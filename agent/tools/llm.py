import os
import json
import threading
import time
from datetime import datetime

from agent.tools.api_config import validate_api_config


OPENAI_COMPATIBLE_MAX_TOKENS = 65536
GPU_SLOT_UNAVAILABLE = "GPU_SLOT_UNAVAILABLE"
LLM_HEARTBEAT_SECONDS = 30
DEFAULT_REQUEST_TIMEOUT_SECONDS = 600


def llm_debug_enabled() -> bool:
    return os.environ.get("AGENT_LLM_DEBUG", "").strip().lower() in {
        "1", "true", "yes", "on",
    }


def _debug_log(message: str) -> None:
    if not llm_debug_enabled():
        return
    timestamp = datetime.now().astimezone().isoformat(timespec="seconds")
    print(f"[{timestamp}] {message}", flush=True)


def _event_log(message: str) -> None:
    print(message, flush=True)


def _content_length(content) -> int:
    if isinstance(content, str):
        return len(content)
    try:
        return len(json.dumps(content, ensure_ascii=False))
    except Exception:
        return len(str(content))


def _prompt_stats(system_prompt: str, messages: list) -> dict:
    system_chars = len(system_prompt)
    message_chars = sum(_content_length(message.get("content", "")) for message in messages)
    total_chars = system_chars + message_chars
    return {
        "system_chars": system_chars,
        "message_chars": message_chars,
        "total_chars": total_chars,
        "rough_tokens": (total_chars + 3) // 4,
        "message_count": len(messages),
    }


def _redacted_error(error: Exception, api_key: str) -> str:
    text = str(error)
    if api_key:
        text = text.replace(api_key, "<redacted>")
    return text


def _is_non_retryable_api_error(error: Exception) -> bool:
    status_code = getattr(error, "status_code", None)
    if status_code in {400, 401, 403, 404, 409, 422}:
        return True
    text = str(error).lower()
    permanent_markers = (
        "context_length_exceeded",
        "maximum context length",
        "max_tokens",
        "invalid_request_error",
        "authentication",
        "invalid api key",
        "permission denied",
        "model_not_found",
    )
    return any(marker in text for marker in permanent_markers)


def _start_heartbeat(label: str, started: float):
    if not llm_debug_enabled():
        return None, None
    stop_event = threading.Event()

    def heartbeat() -> None:
        while not stop_event.wait(LLM_HEARTBEAT_SECONDS):
            elapsed = time.monotonic() - started
            _debug_log(f"⏱️ {label} 仍在等待 API 响应，已等待 {elapsed:.1f} 秒")

    thread = threading.Thread(target=heartbeat, name="llm-api-heartbeat", daemon=True)
    thread.start()
    return stop_event, thread


def _load_role_config(_role: str, api_config_path: str) -> dict:
    with open(api_config_path, "r", encoding="utf-8") as f:
        api_config = json.load(f)
    return validate_api_config(api_config)


def _effective_max_tokens(api_config: dict, requested_max_tokens: int) -> int:
    """Apply provider and endpoint-specific output-token limits."""
    try:
        effective = int(requested_max_tokens)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"max_tokens must be an integer, got {requested_max_tokens!r}"
        ) from exc
    if effective < 1:
        raise ValueError(f"max_tokens must be >= 1, got {effective}")

    configured_limit = api_config.get("max_tokens")
    if configured_limit is not None:
        try:
            configured_limit = int(configured_limit)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"API config max_tokens must be an integer, got {configured_limit!r}"
            ) from exc
        if configured_limit < 1:
            raise ValueError(
                f"API config max_tokens must be >= 1, got {configured_limit}"
            )
        effective = min(effective, configured_limit)

    return min(effective, OPENAI_COMPATIBLE_MAX_TOKENS)


def _execute_call(api_config: dict, system_prompt: str, messages: list, temperature: float, max_tokens: int) -> str:
    api_key = api_config.get("api_key", os.environ.get("API_KEY", ""))
    base_url = api_config.get("base_url", os.environ.get("BASE_URL", ""))
    model_name = api_config.get("model_name", os.environ.get("MODEL_NAME", ""))
    max_tokens = _effective_max_tokens(api_config, max_tokens)

    if not api_key:
        raise ValueError("未配置 API Key；请设置环境变量 API_KEY")

    import openai

    # Accept either API-root form in the user config. The OpenAI-compatible
    # client receives a versioned root even when the user omits /v1.
    if base_url and not base_url.endswith("/v1") and not base_url.endswith("/v1/"):
        base_url = base_url.rstrip("/") + "/v1"
    client_kwargs = {"api_key": api_key, "base_url": base_url}
    if llm_debug_enabled():
        client_kwargs.update(
            timeout=float(api_config.get(
                "request_timeout_seconds", DEFAULT_REQUEST_TIMEOUT_SECONDS
            )),
            max_retries=0,
        )
    client = openai.OpenAI(**client_kwargs)
    full_messages = [{"role": "system", "content": system_prompt}] + messages
    response = client.chat.completions.create(
        model=model_name,
        messages=full_messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return response.choices[0].message.content.strip()


def _ensure_gpu_hold(gpu_holder, role: str) -> None:
    if gpu_holder is None:
        return
    if getattr(gpu_holder, "tensor", None) is not None:
        return
    try:
        occupied = gpu_holder.occupy_on_device()
    except Exception as e:
        raise RuntimeError(f"{GPU_SLOT_UNAVAILABLE}: [{role}] LLM retry failed to occupy GPU holder: {e}") from e
    if not occupied:
        device_id = getattr(gpu_holder, "device_id", "unknown")
        hold_mib = getattr(gpu_holder, "hold_mib", "unknown")
        raise RuntimeError(
            f"{GPU_SLOT_UNAVAILABLE}: [{role}] could not occupy {hold_mib}MiB on GPU {device_id} before LLM retry"
        )


def call_llm(
    role: str,
    system_prompt: str,
    messages: list,
    api_config_path: str,
    temperature: float = 0.3,
    max_tokens: int = 2048,
    gpu_holder=None,
) -> str:
    """
    统一的 LLM 接口，失败时按指数退避重试。
    所有角色从同一个 api.json 读取同一套 OpenAI-compatible API 配置。
    如果传入 gpu_holder，则在每轮尝试和失败 sleep 前确认 holder 仍占住显存。
    """
    api_config = _load_role_config(role, api_config_path)
    debug = llm_debug_enabled()
    if debug:
        stats = _prompt_stats(system_prompt, messages)
        _debug_log(
            f"📦 [{role}] 已加载 API 配置 {api_config_path}; "
            f"messages={stats['message_count']}, system_chars={stats['system_chars']:,}, "
            f"message_chars={stats['message_chars']:,}, total_chars={stats['total_chars']:,}, "
            f"rough_tokens≈{stats['rough_tokens']:,}, requested_max_tokens={max_tokens:,}"
        )
        
    base_cooldown_seconds = 30
    max_cooldown_seconds = 600
    
    def _try_api(config, role_name, round_number):
        provider = "openai"
        model_name = str(config.get("model_name", ""))
        base_url = str(config.get("base_url", ""))
        api_key = str(config.get("api_key", os.environ.get("API_KEY", "")))
        effective_max_tokens = _effective_max_tokens(config, max_tokens)
        request_timeout = float(
            config.get("request_timeout_seconds", DEFAULT_REQUEST_TIMEOUT_SECONDS)
        )
        label = f"[{role_name}] 第 {round_number} 轮 {model_name}"
        _debug_log(
            f"🚀 {label} 开始请求: provider={provider}, base_url={base_url}, "
            f"max_tokens={effective_max_tokens:,}, timeout={request_timeout:g}s"
        )
        started = time.monotonic()
        stop_event, heartbeat_thread = _start_heartbeat(label, started)
        try:
            result = _execute_call(
                config,
                system_prompt,
                messages,
                temperature,
                effective_max_tokens,
            )
            elapsed = time.monotonic() - started
            _debug_log(
                f"✅ {label} 请求成功: elapsed={elapsed:.1f}s, "
                f"response_chars={len(result):,}"
            )
            return result, False
        except Exception as e:
            elapsed = time.monotonic() - started
            non_retryable = _is_non_retryable_api_error(e)
            if debug:
                retry_text = "不可恢复" if non_retryable else "可重试"
                _debug_log(
                    f"⚠️ {label} 调用失败 ({retry_text}): elapsed={elapsed:.1f}s, "
                    f"error_type={type(e).__name__}, error={_redacted_error(e, api_key)}"
                )
            else:
                _event_log(
                    f"⚠️ [{role_name}] API 调用失败: "
                    f"{_redacted_error(e, api_key)}"
                )
            return None, non_retryable
        finally:
            if stop_event is not None:
                stop_event.set()
            if heartbeat_thread is not None:
                heartbeat_thread.join(timeout=0.2)

    round_failures = 0
    while True:
        _ensure_gpu_hold(gpu_holder, role)
        round_number = round_failures + 1

        result, non_retryable = _try_api(api_config, role, round_number)
        if result is not None:
            return result

        if debug and non_retryable:
            raise RuntimeError(
                f"[{role}] API 返回不可恢复错误；停止重试。"
            )

        round_failures += 1
        backoff_power = min(round_failures - 1, 5)
        cooldown_seconds = min(max_cooldown_seconds, base_cooldown_seconds * (2 ** backoff_power))
        _event_log(f"⏳ [{role}] API 本轮失败，{cooldown_seconds} 秒后重试")
        _ensure_gpu_hold(gpu_holder, role)
        remaining = cooldown_seconds
        while remaining > 0:
            step = min(LLM_HEARTBEAT_SECONDS, remaining)
            time.sleep(step)
            remaining -= step
            if debug and remaining > 0:
                _debug_log(f"⏳ [{role}] 退避等待中，距离下一轮还有 {remaining} 秒")
