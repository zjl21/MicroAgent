"""
train.py — 一键训练入口

用法::

    python train.py --config config.yaml
"""

import argparse
import os
import warnings

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)
os.environ.setdefault("PYTHONWARNINGS", "ignore")
warnings.filterwarnings("ignore")

import torch
import yaml

from toolbox.utility.paths import set_project_root
from toolbox.utility.iolib import load_config

set_project_root(os.path.dirname(__file__))

from agent.runtime import Trainer, build_model, build_step_fn, build_volume
from agent.tools.gpu_holder import GPUHolder

# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="微结构定量训练入口")
    parser.add_argument(
        "--config", default="experiments/exp-slice/config.yaml",
        help="path of config file",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)

    # ── 环境─────────────────────────────────────────────────────
    env_cfg = cfg.get("env", {})
    gpu_id  = int(env_cfg.get("gpu_id", "0"))
    device_str = env_cfg.get("device", "cuda")
    if device_str == "cuda" and torch.cuda.is_available():
        if os.environ.get("CUDA_VISIBLE_DEVICES") is not None:
            device = torch.device("cuda:0")
        else:
            device = torch.device(f"cuda:{gpu_id}")
            torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")
    torch.manual_seed(env_cfg.get("seed", 42))
    GPUHolder.setup_process_memory(
        device,
        limit_mib=env_cfg.get("gpu_memory_limit_mib"),
        reserve_cache=env_cfg.get("gpu_reserve_cache", True),
        required=True,
    )

    # ── 数据：体积采样器 ───────────────────────────────────────────────
    volume = build_volume(cfg, device)
    n_grad = volume.bval.shape[0]
    cfg["data"]["n_grad"] = n_grad
    with open(args.config, 'w') as f:
        yaml.dump(cfg, f)

    # ── 模型 & 损失 ────────────────────────────────────────────────────────
    model   = build_model(cfg)
    step_fn = build_step_fn(cfg, volume.bval, volume.bvec, volume.norm_factor,
                            volume=volume, device=device)

    # ── 训练 ──────────────────────────────────────────────────────────────
    trainer = Trainer(model, volume, step_fn, cfg, cfg_path=args.config)
    trainer.train()

    del trainer, model, volume
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
