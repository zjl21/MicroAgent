import os
import json
import math
import torch
import random
import gc

from toolbox.utility.paths import set_project_root
from toolbox.utility.iolib import load_config

set_project_root()
from agent.runtime import Trainer, build_volume, build_model, build_step_fn
from agent.tools.gpu_holder import GPUHolder


# ── 代码层硬判断：无需 LLM ────────────────────────────────────────────────

def has_nan_or_inf(history: list) -> bool:
    """检查训练日志中是否出现过 NaN / Inf 损失值。"""
    for entry in history:
        for key in ("train_loss", "val_loss"):
            v = entry.get(key)
            if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
                return True
    return False


def is_loss_flat(history: list, threshold: float = 0.05, window: int = 5) -> bool:
    """
    检查 val_loss 是否陷入平坦。
    1. 整体从头到尾几乎没有下降：(首轮 - 最低) / 首轮 < threshold
    2. 或者末尾 window 轮内几乎没有变化：(局部最大 - 局部最小) / 局部最大 < 1e-3
    """
    losses = [e["val_loss"] for e in history if isinstance(e.get("val_loss"), (int, float))]
    if len(losses) < 2:
        return False
        
    first, best = losses[0], min(losses)
    # 情况1：整体下降幅度极小
    if first > 0 and (first - best) / first < threshold:
        return True
        
    # 情况2：虽然前期有下降，但末尾完全陷入停滞（如梯度消失直接不更新了）
    if len(losses) >= window:
        recent = losses[:window]
        r_max, r_min = max(recent), min(recent)
        if r_max > 0 and (r_max - r_min) / r_max < 1e-3:
            return True
            
    return False


# ── 梯度采集 ──────────────────────────────────────────────────────────────

def grad_stats(model) -> dict:
    """收集每个有梯度的参数层的 grad norm / mean / max。"""
    stats = {}
    for name, p in model.named_parameters():
        if p.grad is None:
            stats[name] = {"norm": None, "mean": None, "max": None}
        else:
            g = p.grad.detach().float()
            stats[name] = {
                "norm": round(g.norm().item(), 8),
                "mean": round(g.abs().mean().item(), 8),
                "max":  round(g.abs().max().item(), 8),
            }
    return stats


# ── 梯度诊断主函数 ────────────────────────────────────────────────────────

def run_diagnose_grad(config_path: str, gpu_holder=None) -> str:
    """
    前向 + 反向传播 steps 次，采集各层梯度统计，写入 JSON 报告。
    返回报告文件路径。
    """
    cfg = load_config(config_path)
    env_cfg = cfg.get("env", {})
    device_name = env_cfg.get("device", "cuda")
    gpu_id = env_cfg.get("gpu_id", None)
    if gpu_id is None and gpu_holder is not None:
        gpu_id = getattr(gpu_holder, "device_id", 0)
    gpu_id = int(gpu_id or 0)

    if device_name == "cuda" and torch.cuda.is_available():
        device = torch.device(f"cuda:{gpu_id}")
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")
    GPUHolder.setup_process_memory(
        device,
        limit_mib=env_cfg.get("gpu_memory_limit_mib"),
        reserve_cache=env_cfg.get("gpu_reserve_cache", True),
        required=True,
    )

    exp_dir = os.path.dirname(os.path.abspath(config_path))
    out_path = os.path.join(exp_dir, "grad_diagnosis.json")

    volume = build_volume(cfg, device)
    if volume is None:
        raise ValueError("未能成功构建训练 volume，梯度诊断失败。")

    n_grad = volume.bval.shape[0]
    cfg["data"]["n_grad"] = n_grad

    model   = build_model(cfg)
    step_fn = build_step_fn(cfg, volume.bval, volume.bvec, volume.norm_factor,
                            volume=volume, device=device)
    trainer = Trainer(model, volume, step_fn, cfg, cfg_path=config_path)

    batch_size = cfg["data"].get("batch_size", 2048)
    ids_arr = list(volume.train_ids)
    random.shuffle(ids_arr)

    log_data = []
    for i in range(len(ids_arr) // batch_size + 1):
        start = i * batch_size
        if start >= len(ids_arr):
            break
        batch = volume.sample_batch(ids_arr[start:start + batch_size])
        batch = trainer._augment(batch)

        model.train()
        trainer.optimizer.zero_grad()
        loss , _ = step_fn(model, batch, device)
        loss.backward()

        log_data.append({
            "step": i,
            "loss": loss.item(),
            "grad_stats": grad_stats(model),
        })

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(log_data, f, indent=2, ensure_ascii=False)

    del trainer, model, volume
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return out_path
