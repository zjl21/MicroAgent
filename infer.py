"""
infer.py — 推理入口

从 checkpoint 加载模型，对整个体积（train + val 切片/体素）运行推理，
输出 DTI 参数图（S0、tensor、FA、MD 等）。

用法::

    python infer.py --config config.yaml
"""

import argparse
import os
import warnings

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)
os.environ.setdefault("PYTHONWARNINGS", "ignore")
warnings.filterwarnings("ignore")

import torch

from toolbox.utility.paths import set_project_root
from toolbox.utility.iolib import load_config

set_project_root(os.path.dirname(__file__))

from agent.runtime import build_dataset, build_model, infer_volume
from agent.tools.gpu_holder import GPUHolder


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="微结构定量推理入口")
    parser.add_argument(
        "--config", default="experiments/exp-slice/config.yaml",
        help="config path",
    )
    args = parser.parse_args()
    cfg = load_config(args.config)
    exp_dir = os.path.dirname(os.path.abspath(args.config))
    ckpt_path = os.path.join(exp_dir, 'checkpoints', "latest_model.pth")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    # ── 环境 ──────────────────────────────────────────────────────────────
    env_cfg    = cfg.get("env", {})
    gpu_id     = int(env_cfg.get("gpu_id", "0"))
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

    # ── 数据（不做 train/val 分割，推理全部切片）─────────────────────────
    seed   = env_cfg.get("seed", 42)
    volume = build_dataset(cfg, device=device, preload_gpu=False, seed=seed, split=False)
    cfg["data"]["n_grad"] = volume.bval.shape[0]

    # ── 模型 ──────────────────────────────────────────────────────────────
    model = build_model(cfg, device)
    # PyTorch 2.6 defaults weights_only=True, which can reject older trusted checkpoints
    # that were saved with optimizer/scheduler objects in the pickle stream.
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])

    # ── 推理并保存 ──────────────────────────────────────────────────────────────

    out_dir = os.path.join(exp_dir, "output")
    os.makedirs(out_dir, exist_ok=True)
    infer_volume(cfg, model, volume, out_dir)

if __name__ == "__main__":
    main()
