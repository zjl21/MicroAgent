import os
import json
import yaml

from agent.tools.llm import call_llm, llm_debug_enabled
from agent.tools.utility import extract_tag, get_all_yaml_keys
from agent.prompts.architect import ArchitectPrompt
from agent.runtime import config_reference_path
from constrain import build_constraint_context, validate_config_constraints


class Architect:

    def __init__(self, task_dir: str, hard_requirements: dict = None,
                 max_trials: int = 20, skill: list = None, test: bool = False,
                 resume_dir: str = None, quiet: bool = False, gpu_holder=None,
                 use_successful_experience: bool = False,
                 physics_model: str = None):
        self.task_dir = task_dir
        self.api_config_path = os.path.join(os.path.dirname(task_dir), "api.json")
        self.hard_requirements = hard_requirements or {}
        self.skill = skill or []
        self.reference_path = config_reference_path()
        self.quiet = quiet
        self.gpu_holder = gpu_holder
        self.use_successful_experience = use_successful_experience
        self.physics_model = physics_model
        self.required_model_family = _hard_requirement_value(
            self.hard_requirements, "model.name"
        )
        self.reference_content = None
            
        with open(self.reference_path, "r", encoding="utf-8") as f:
            reference_content = f.read()

        if not test:
            self.constraint_context = build_constraint_context(self.task_dir)
            self.past_records = []
            self.system_prompt = ArchitectPrompt.build_system(
                reference_content,
                os.path.dirname(self.task_dir),
                max_trials,
                skill=self.skill,
                use_successful_experience=self.use_successful_experience,
                model=self.physics_model,
                target_features_path=self.constraint_context.get("features_path"),
                required_model_family=self.required_model_family,
            )
        else:
            self.resume_dir = resume_dir
            self.constraint_context = build_constraint_context(self.task_dir, test=True)
            self.system_prompt = ArchitectPrompt.build_system_test(
                reference_content,
                self.resume_dir,
                skill=self.skill,
                target_features_path=self.constraint_context.get("features_path"),
                required_model_family=self.required_model_family,
            )
            self.reference_content = reference_content

    # ── 持久化 ────────────────────────────────────────────────────────────

    def save(self, path: str):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.past_records, f, ensure_ascii=False, indent=2)

    def load(self, path: str):
        with open(path, "r", encoding="utf-8") as f:
            self.past_records = json.load(f)

    # ── 公开接口 ──────────────────────────────────────────────────────────

    def design(self, trial_dir: str,
               summary: str = "", diagnosis: str=""):
        """生成配置文件，保存 thought 和 config，并更新内部记忆。"""
        trial_num = int(os.path.basename(trial_dir).split("_")[-1])
        task_idx = int(os.path.dirname(trial_dir).split('_')[-1])
        user_prompt = ArchitectPrompt.build_user(self.task_dir, task_idx, self.hard_requirements, self.past_records, summary, diagnosis)
        thought_str, config_str, issue_str = self._call_llm(user_prompt)

        with open(os.path.join(trial_dir, "architect_thought.md"), "w", encoding="utf-8") as f:
            f.write(thought_str)
        with open(os.path.join(trial_dir, "config.yaml"), "w", encoding="utf-8") as f:
            f.write(config_str)
        if issue_str:
            with open(os.path.join(trial_dir, "architect_issue.md"), "w", encoding="utf-8") as f:
                f.write(issue_str)
        if not self.quiet:
            print(f"✨ 配置文件及思维链已生成并保存至: {trial_dir}", flush=True)

        self.past_records.append({
            "trial": trial_num,
            "query": summary if summary else user_prompt,
            "thought": thought_str,
            "config": config_str,
        })

    def design_test(self, trial_dir: str, model: str):
        """在测试任务上生成配置文件"""
        self.system_prompt = ArchitectPrompt.build_system_test(
            self.reference_content,
            self.resume_dir,
            skill=self.skill,
            use_successful_experience=self.use_successful_experience,
            model=model,
            target_features_path=self.constraint_context.get("features_path"),
            required_model_family=self.required_model_family,
        )
        user_prompt = ArchitectPrompt.build_user_test(self.constraint_context["features_path"], model, self.hard_requirements)
        thought_str, config_str, issue_str = self._call_llm(user_prompt)

        with open(os.path.join(trial_dir, "architect_thought.md"), "w", encoding="utf-8") as f:
            f.write(thought_str)
        with open(os.path.join(trial_dir, "config.yaml"), "w", encoding="utf-8") as f:
            f.write(config_str)
        if issue_str:
            with open(os.path.join(trial_dir, "architect_issue.md"), "w", encoding="utf-8") as f:
                f.write(issue_str)
        if not self.quiet:
            print(f"✨ 配置文件及思维链已生成并保存至: {trial_dir}", flush=True)

    # ── 私有方法 ──────────────────────────────────────────────────────────

    def _call_llm(self, user_prompt: str) -> tuple:
        '''调用大模型，解析其结果'''
        
        ref_dict = get_all_yaml_keys(self.reference_path)

        debug = llm_debug_enabled()
        if not self.quiet and debug:
            system_chars = len(self.system_prompt)
            user_chars = len(user_prompt)
            total_chars = system_chars + user_chars
            print(
                "呼叫大模型执行架构师(Architect)任务: "
                f"system_chars={system_chars:,}, user_chars={user_chars:,}, "
                f"total_chars={total_chars:,}, rough_tokens≈{(total_chars + 3) // 4:,}",
                flush=True,
            )
        elif not self.quiet:
            print("呼叫大模型执行架构师(Architect)任务中，请稍候...", flush=True)

        max_retries = 50
        messages = [{"role": "user", "content": user_prompt}]
        thought_parts, config_str, issue_parts = [], None, []

        for attempt in range(max_retries):
            if not self.quiet and debug:
                message_chars = sum(len(str(item.get("content", ""))) for item in messages)
                print(
                    f"🧭 Architect 输出尝试 {attempt + 1}/{max_retries}: "
                    f"messages={len(messages)}, message_chars={message_chars:,}",
                    flush=True,
                )
            output_text = call_llm(
                role="architect",
                system_prompt=self.system_prompt,
                messages=messages,
                api_config_path=self.api_config_path,
                temperature=0.2,
                max_tokens=100000,
                gpu_holder=self.gpu_holder,
            )

            if not self.quiet and debug:
                print(
                    f"📨 Architect 收到响应: chars={len(output_text):,}, "
                    f"has_thought={bool(extract_tag(output_text, 'thought'))}, "
                    f"has_config={bool(extract_tag(output_text, 'config'))}, "
                    f"has_issue={bool(extract_tag(output_text, 'issue'))}",
                    flush=True,
                )

            thought_str = extract_tag(output_text, "thought")
            if thought_str:
                thought_parts.append(thought_str.strip())
            config_str  = extract_tag(output_text, "config")
            issue_str   = extract_tag(output_text, "issue")
            if issue_str:
                issue_parts.append(issue_str.strip())

            if not config_str:
                if not self.quiet and debug:
                    print("⚠️ Architect 响应缺少有效的 <config> 标签，要求模型重试", flush=True)
                messages.append({"role": "assistant", "content": output_text})
                messages.append({"role": "user", "content": f"{ArchitectPrompt.NOT_CONFIG}"})
                continue

            # 清理 markdown 标签
            if config_str.startswith("```yaml"):
                config_str = config_str[7:]
            elif config_str.startswith("```"):
                config_str = config_str[3:]
            if config_str.endswith("```"):
                config_str = config_str[:-3]

            try:
                gen_dict = yaml.safe_load(config_str)
                is_valid, msg = _validate_config_keys(gen_dict, ref_dict)
                if is_valid:
                    is_valid, msg = validate_config_constraints(gen_dict, self.constraint_context)
                if is_valid:
                    is_valid, msg = validate_hard_requirements(
                        gen_dict, self.hard_requirements
                    )
                if not is_valid:
                    if not self.quiet:
                        print(f"⚠️ 架构师乱造参数，已拦截: {msg}", flush=True)
                    messages.append({"role": "assistant", "content": output_text})
                    messages.append({"role": "user", "content": f"{msg}\n{ArchitectPrompt.NOT_VALID}"})
                    continue
                else:
                    if not self.quiet and debug:
                        print("✅ Architect 配置通过 YAML、字段和约束校验", flush=True)
                    return "\n\n".join(thought_parts), config_str, "\n\n".join(issue_parts) or None
            except Exception as e:
                if not self.quiet:
                    print(f"⚠️ 架构师生成的 YAML 语法破损，已拦截: {str(e)}", flush=True)
                messages.append({"role": "assistant", "content": output_text})
                messages.append({"role": "user", "content": f"{str(e)}\n{ArchitectPrompt.NOT_YAML}"})
                continue

        raise RuntimeError("架构师(Architect)连续多次生成无效配置，已终止尝试。")


# ── 辅助函数 ──────────────────────────────────────────────────────────────

_MISSING = object()


def _hard_requirement_value(hard_requirements, dotted_key, default=None):
    items = hard_requirements if isinstance(hard_requirements, list) else [hard_requirements]
    for item in items:
        if isinstance(item, dict) and dotted_key in item:
            return item[dotted_key]
    return default


def _nested_value(config, dotted_key):
    value = config
    for key in dotted_key.split("."):
        if not isinstance(value, dict) or key not in value:
            return _MISSING
        value = value[key]
    return value


def _same_required_value(actual, expected):
    if isinstance(expected, bool):
        return isinstance(actual, bool) and actual is expected
    return actual == expected


def validate_hard_requirements(config, hard_requirements):
    """Enforce structured dotted-path requirements after LLM generation."""
    items = hard_requirements if isinstance(hard_requirements, list) else [hard_requirements]
    for item in items:
        if not isinstance(item, dict):
            continue
        for dotted_key, expected in item.items():
            actual = _nested_value(config, dotted_key)
            if actual is _MISSING:
                return False, f"Required field '{dotted_key}' is missing."
            if not _same_required_value(actual, expected):
                return False, (
                    f"Hard requirement violated: '{dotted_key}' must equal "
                    f"{expected!r}, got {actual!r}."
                )
    return True, "Valid"

def _validate_config_keys(gen_config, ref_config, path=""):
    """递归检查 gen_config 的 keys，如果发现不存在于 ref_config 则返回错。"""
    for k, v in gen_config.items():
        if k not in ref_config:
            return False, f"Invalid field: '{path}.{k}' is not in the reference dictionary! If you want to use this feature, please follow the existing structure in config_reference.yaml."
        if isinstance(v, dict) and isinstance(ref_config[k], dict):
            is_valid, msg = _validate_config_keys(v, ref_config[k], f"{path}.{k}" if path else k)
            if not is_valid:
                return False, msg
    return True, "Valid"
