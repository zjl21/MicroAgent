import json
import os
import re
import copy
import yaml
from yaml.events import MappingStartEvent, MappingEndEvent, ScalarEvent, SequenceStartEvent, SequenceEndEvent
from yaml import parse
import math

from toolbox.utility.iolib import load_config


def normalize_hard_requirements(hard_requirements):
    if not hard_requirements:
        return []
    if isinstance(hard_requirements, dict):
        return [hard_requirements.copy()]
    if isinstance(hard_requirements, str):
        return [hard_requirements]
    if isinstance(hard_requirements, (list, tuple)):
        normalized = []
        for item in hard_requirements:
            if isinstance(item, dict):
                normalized.append(item.copy())
            elif isinstance(item, str):
                normalized.append(item)
            else:
                raise TypeError(f"Unsupported hard requirement type: {type(item).__name__}")
        return normalized
    raise TypeError(f"Unsupported hard requirements type: {type(hard_requirements).__name__}")


def merge_hard_requirement_dict(hard_requirements, extra_requirements: dict):
    hard_requirements = normalize_hard_requirements(hard_requirements)
    for item in hard_requirements:
        if isinstance(item, dict):
            item.update(extra_requirements)
            return hard_requirements
    hard_requirements.append(extra_requirements.copy())
    return hard_requirements


def write_json_atomic(path, data):
    tmp_path = f"{path}.tmp.{os.getpid()}"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(tmp_path, path)


def ensure_symlink(src, dst):
    if os.path.lexists(dst):
        return
    os.symlink(src, dst)


def build_task_hard_requirements(base_hard_requirements, task, data_dir, gpu_id, task_prompt, memory_limit_mib=None):
    task_requirements = copy.deepcopy(base_hard_requirements)
    routing_requirements = {
        "data.data_dir": data_dir,
        "env.gpu_id": gpu_id,
    }
    if memory_limit_mib is not None:
        routing_requirements["env.gpu_memory_limit_mib"] = int(memory_limit_mib)
    task_requirements = merge_hard_requirement_dict(task_requirements, routing_requirements)
    task_requirements = merge_hard_requirement_dict(
        task_requirements,
        task_prompt.build_task_hard_requirements(task),
    )
    return task_requirements


def rewrite_config_gpu(output_path, gpu_id, memory_limit_mib=None):
    cfg = load_config(output_path)
    cfg.setdefault("env", {})
    gpu_idx_cfg = cfg.get("env", {}).get("gpu_id", gpu_id)
    needs_update = str(gpu_idx_cfg) != str(gpu_id)
    if needs_update:
        cfg["env"]["gpu_id"] = gpu_id
    if memory_limit_mib is not None and cfg["env"].get("gpu_memory_limit_mib") != int(memory_limit_mib):
        cfg["env"]["gpu_memory_limit_mib"] = int(memory_limit_mib)
        needs_update = True
    if needs_update:
        with open(output_path, "w", encoding="utf-8") as f:
            yaml.dump(cfg, f)
    return cfg


def extract_tag(text, tag):
    pattern = f"<{tag}>(.*?)</{tag}>"
    match = re.search(pattern, text, re.DOTALL)
    return match.group(1).strip() if match else ""

def get_all_yaml_keys(filepath):
    """
    Parses a yaml file and aggregates all keys.
    """
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()

    allowed_schema = {}
    
    try:
        events = list(parse(content))
    except Exception as e:
        raise RuntimeError(f"无法解析 YAML 文件: {e}")
        
    stack = [allowed_schema]
    current_key = None
    in_sequence = 0
    
    for i, event in enumerate(events):
        if isinstance(event, MappingStartEvent):
            if current_key is not None:
                if current_key not in stack[-1]:
                    stack[-1][current_key] = {}
                elif not isinstance(stack[-1][current_key], dict):
                    stack[-1][current_key] = {}
                stack.append(stack[-1][current_key])
                current_key = None
            else:
                pass
                
        elif isinstance(event, MappingEndEvent):
            if len(stack) > 1:
                stack.pop()
                
        elif isinstance(event, SequenceStartEvent):
            if current_key is not None:
                stack[-1][current_key] = []
                current_key = None
            in_sequence += 1
            
        elif isinstance(event, SequenceEndEvent):
            in_sequence -= 1
                
        elif isinstance(event, ScalarEvent):
            if in_sequence > 0:
                pass # ignore sequence items
            else:
                if current_key is None:
                    current_key = event.value
                    if current_key not in stack[-1]:
                        stack[-1][current_key] = None
                else:
                    current_key = None
                
    return allowed_schema

def compress_history(history: list, max_points: int = 100, context_window: int = 3) -> list:
    """
    将完整训练日志压缩为语义上重要的节点及其上下文，供 LLM 分析。

    先找出所有语义关键点，再为每个关键点保留前后 context_window 条记录，
    最后均匀采样补足至 max_points。

    关键点类型：
        1. 前 5 条 + 后 5 条
        2. val_loss 全局最优点
        3. val_loss 拐点（平滑后斜率符号变化 / 斜率突变）
        4. train/val loss 分歧开始的点（gap 开始持续扩大）
    """
    n = len(history)
    if n <= max_points:
        return history

    val_losses   = [e.get("val_loss",   float("nan")) for e in history]
    train_losses = [e.get("train_loss", float("nan")) for e in history]

    anchor_idx = set()

    # 1. 首尾
    anchor_idx.update(range(min(5, n)))
    anchor_idx.update(range(max(0, n - 5), n))

    # 2. val_loss 全局最优
    valid_val = [(i, v) for i, v in enumerate(val_losses)
                    if isinstance(v, float) and not math.isnan(v)]
    if valid_val:
        anchor_idx.add(min(valid_val, key=lambda x: x[1])[0])

    # 3. val_loss 拐点（基于平滑斜率）
    smooth_window = max(3, n // 50)
    smoothed = []
    for i in range(n):
        lo = max(0, i - smooth_window // 2)
        hi = min(n, i + smooth_window // 2 + 1)
        vals = [v for v in val_losses[lo:hi]
                if isinstance(v, float) and not math.isnan(v)]
        smoothed.append(sum(vals) / len(vals) if vals else float("nan"))

    slopes = [float("nan")] + [
        smoothed[i] - smoothed[i - 1]
        if not (math.isnan(smoothed[i]) or math.isnan(smoothed[i - 1]))
        else float("nan")
        for i in range(1, n)
    ]


    # 斜率突变（超过 3σ）
    valid_slopes = [s for s in slopes if not math.isnan(s)]
    if valid_slopes:
        mean_s = sum(valid_slopes) / len(valid_slopes)
        std_s  = (sum((s - mean_s) ** 2 for s in valid_slopes) / len(valid_slopes)) ** 0.5
        for i in range(1, n - 1):
            if math.isnan(slopes[i]) or math.isnan(slopes[i - 1]):
                continue
            if abs(slopes[i] - slopes[i - 1]) > 3 * std_s:
                anchor_idx.add(i)

    # 4. train/val 分歧点：gap = val - train 开始持续扩大的起点
    valid_gaps = [
        (i, val_losses[i] - train_losses[i])
        for i in range(n)
        if isinstance(val_losses[i], float) and isinstance(train_losses[i], float)
        and not math.isnan(val_losses[i]) and not math.isnan(train_losses[i])
    ]
    if len(valid_gaps) > 10:
        min_gap_i, min_gap_v = min(valid_gaps, key=lambda x: x[1])
        anchor_idx.add(min_gap_i)
        threshold = min_gap_v + max(abs(min_gap_v) * 0.5, 1e-4)
        for i, g in valid_gaps:
            if i > min_gap_i and g > threshold:
                anchor_idx.add(i)
                break

    # 每个 anchor 扩展为前后 context_window 的窗口
    keep_idx = set()
    for a in anchor_idx:
        for offset in range(-context_window, context_window + 1):
            idx = a + offset
            if 0 <= idx < n:
                keep_idx.add(idx)

    # 均匀采样补足
    remaining = max_points - len(keep_idx)
    if remaining > 0:
        step = n / (remaining + 1)
        for k in range(1, remaining + 1):
            keep_idx.add(int(k * step))

    return [history[i] for i in sorted(keep_idx)]
