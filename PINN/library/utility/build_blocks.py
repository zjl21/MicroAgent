import torch
import json
import os
from glob import glob
from library.dataio.diffusion import build_dataset
import library.losses.data_consistency as dc_losses
import library.losses.parameter_map as pm_losses

from toolbox.utility.iolib import load_config

#  ---------------------------------------------------------------------------
# 数据集工厂
# ---------------------------------------------------------------------------

def build_volume(cfg: dict, device: torch.device):
    """构建数据集（voxelwise 或 slicewise），由 cfg.env.preload_gpu 决定是否预加载到 GPU。"""
    seed        = cfg.get("env", {}).get("seed", 42)
    preload_gpu = cfg.get("env", {}).get("preload_gpu", True)
    return build_dataset(cfg, device=device, preload_gpu=preload_gpu, seed=seed)


# ---------------------------------------------------------------------------
# 模型工厂
# ---------------------------------------------------------------------------

def build_model(cfg: dict, device=None) -> torch.nn.Module:
    model_cfg   = cfg["model"]
    family_name = model_cfg["name"]

    # out_params 从 physics 类读取，不在 config 里硬编码
    physics_name = cfg["physics"]["name"]
    try:
        physics_module = __import__(f"library.physics.{physics_name}", fromlist=[physics_name])
        physics_class  = getattr(physics_module, physics_name)
        out_params     = physics_class.N_PARAMS
    except Exception as e:
        raise ValueError(f"Failed to get N_PARAMS from physics '{physics_name}': {e}")

    dataset_type = cfg["data"].get("dataset", "patchwise")
    spatial_dim = {"patchwise": 3, "slicewise": 2}.get(dataset_type)

    model_routes = {
        "voxelwise": ("voxelwise", "MLP"),
        "coords": ("coords", "CoordINR"),
        "spatial": ("spatial", "SpatialModel"),
        "token": ("token", "TokenModel"),
        "window_attention": ("window_attention", "WindowAttentionModel"),
        "operator": ("operator", "OperatorModel"),
    }
    if family_name not in model_routes:
        raise ValueError(
            f"Unknown model family '{family_name}'. "
            f"Options: {tuple(model_routes)}"
        )

    if family_name in ("spatial", "token", "window_attention", "operator"):
        if spatial_dim is None:
            raise ValueError(f"model.name='{family_name}' requires slicewise or patchwise data, not '{dataset_type}'.")
    elif family_name == "voxelwise":
        if dataset_type != "voxelwise":
            raise ValueError(f"model.name='voxelwise' requires voxelwise data, not '{dataset_type}'.")
    elif family_name == "coords":
        if dataset_type not in ("3Dcoord", "coords"):
            raise ValueError(f"model.name='coords' requires 3Dcoord/coords data, not '{dataset_type}'.")
    else:
        raise ValueError(f"Unknown dataset type for model build: '{dataset_type}'")

    module_name, class_name = model_routes[family_name]
    model_module = __import__(f"library.models.{module_name}", fromlist=[class_name])
    model_class = getattr(model_module, class_name)

    model_kwargs = model_cfg.get(family_name, {})
    if not isinstance(model_kwargs, dict):
        raise ValueError(f"model.{family_name} must be a dictionary.")
    kwargs = {k: v for k, v in model_kwargs.items() if k not in ("method", "name", "type")}
    build_kwargs = {
        "n_grad": cfg["data"]["n_grad"],
        "out_params": out_params,
        **kwargs,
    }
    if spatial_dim is not None:
        build_kwargs["spatial_dim"] = spatial_dim

    kl_enabled = cfg.get("loss", {}).get("KL", {}).get("weight", 0.0) > 0.0
    if family_name in ("coords", "spatial", "token", "window_attention", "operator"):
        build_kwargs["use_VAE"] = bool(build_kwargs.get("use_VAE", False) or kl_enabled)
    model = model_class(**build_kwargs)
    return model.to(device) if device is not None else model


# ---------------------------------------------------------------------------
# shell_weights 工厂
# ---------------------------------------------------------------------------

def _build_shell_weights(shell_weighting: str, dc_cfg: dict, volume) -> torch.Tensor | None:
    """
    根据 shell_weighting 模式构造 (N,) 权重 tensor，归一化到均值为 1。
    返回 None 表示不加权。
    """
    if shell_weighting == "none":
        return None

    bval   = volume.bval.float()
    shells = bval.unique(sorted=True)

    if shell_weighting == "inv_bval":
        bval_w = bval.clone()
        bval_w[bval_w == 0] = 1.0   # b0: weight = 1/1 = 1
        w = 1.0 / bval_w

    elif shell_weighting == "inv_log_bval":
        bval_w = bval.clone()
        w = 1.0 / torch.log(bval_w)
        w[bval_w == 0] = 1.0

    elif shell_weighting == "inv_sqrt_bval":
        bval_w = bval.clone()
        bval_w[bval_w == 0] = 1.0
        w = 1.0 / torch.sqrt(bval_w)

    elif shell_weighting == "inv_signal":
        sig_w = volume.shell_mean_signal.float().clamp(min=1e-6)
        w = 1.0 / sig_w

    elif shell_weighting == "inv_volume_num":
        # weight = 1 / (该 shell 的方向数)，方向多的 shell 每个方向贡献更小
        w = torch.zeros(len(bval))
        for s in shells:
            idx = (bval == s)
            w[idx] = 1.0 / idx.sum().float()

    elif shell_weighting == "manual":
        manual = dc_cfg.get("shell_weights", [])
        if not manual:
            raise ValueError("shell_weighting='manual' requires 'shell_weights' list in config")
        if len(manual) != len(shells):
            raise ValueError(
                f"shell_weights has {len(manual)} entries but data has {len(shells)} shells"
            )
        w = torch.zeros(len(bval))
        for i, s in enumerate(shells):
            w[bval == s] = float(manual[i])

    else:
        print(f"Unknown shell_weighting: '{shell_weighting}'")
        return None

    w = w / w.mean()   # 归一化：保持 loss 量级不变
    return w.to(volume.device)


def _build_s_noise(cfg: dict, value, volume, device) -> torch.Tensor:
    """
    Rician sigma in the same normalized signal domain as training data.

    Numeric config values are used directly. "auto"/None estimates sigma from
    data_dir/*_features.json and rescales it by volume.norm_factor.
    """
    if value in (None, "auto"):
        snr, b0_mean = None, None
        data_dir = cfg.get("data", {}).get("data_dir")
        if data_dir:
            feature_files = glob(os.path.join(data_dir, "*_features.json"))
            if feature_files:
                try:
                    with open(feature_files[0], "r", encoding="utf-8") as f:
                        signal_features = json.load(f).get("signal", {})
                    snr = signal_features.get("SNR")
                    b0_stats = signal_features.get("b0", {})
                    b0_mean = b0_stats.get("mean")
                except Exception:
                    snr = None

        if snr and float(snr) > 0:
            sigma_raw = float(b0_mean) / float(snr) if b0_mean else None
            norm_factor = volume.norm_factor
            if hasattr(norm_factor, "item"):
                norm_factor = norm_factor.item()
            if sigma_raw and norm_factor:
                value = sigma_raw / float(norm_factor)
            else:
                value = 1.0 / float(snr)
        else:
            value = 0.05

    tensor = torch.as_tensor(value, dtype=torch.float32, device=device or volume.device)
    return tensor.clamp_min(1e-8)


def _shell_label(value):
    value = float(value)
    if abs(value - round(value)) < 1e-4:
        return str(int(round(value)))
    return f"{value:g}".replace(".", "p")


# ---------------------------------------------------------------------------
# model input 工具
# ---------------------------------------------------------------------------

def _model_input_key(cfg: dict) -> str:
    model_name = cfg.get("model", {}).get("name")
    if model_name == "coords":
        return "coords"
    return "signal"


# ---------------------------------------------------------------------------
# step_fn 工厂
# ---------------------------------------------------------------------------

def build_step_fn(cfg: dict, bval, bvec, norm_factor, volume, device=None):
    """根据 config 构建单步损失函数。

    physics.output_normalize = true 时：
        网络输出任意实数 → sigmoid → [0,1] → physics_class.denormalize() → 物理范围
    physics.output_normalize = false（默认）：
        网络输出直接作为物理参数传入前向模型
    """
    name             = cfg["physics"]["name"]
    # output_normalize = cfg["physics"].get("output_normalize", False)

    try:
        physics_module   = __import__(f"library.physics.{name}", fromlist=[name])
        physics_class    = getattr(physics_module, name)
        physics_instance = physics_class(bval, bvec, norm_factor)
        require_S0       = getattr(physics_class, "require_S0", False)
    except Exception as e:
        raise ValueError(f"Unknown physics model: {name}")

    # 一次性把物理模型的固定 tensor 搬到目标设备，避免 step_fn 里每步都做
    if device is not None and hasattr(physics_instance, 'gradient'):
        physics_instance.gradient = physics_instance.gradient.to(device)

    # ── 提前解析并收集损失函数对象，避免在 step_fn 中每次判断 ─────────────────
    loss_cfg = cfg.get("loss", {})
    kl_cfg = loss_cfg.get("KL", {})
    kl_weight = float(kl_cfg.get("weight", 0.0) or 0.0)
    kl_beta = float(kl_cfg.get("beta", 1.0) or 1.0)
    use_kl = kl_weight > 0.0

    weight = cfg['physics'].get('constrain_weight', 0.0)
    if weight > 0.0:
        if "parameter_map" not in loss_cfg or not isinstance(loss_cfg["parameter_map"], dict):
            loss_cfg["parameter_map"] = {}
        loss_cfg["parameter_map"]["constrain"] = {"weight": weight}

    pm_loss_fns = []
    if "parameter_map" in loss_cfg and loss_cfg["parameter_map"]:
        pm_dict = loss_cfg["parameter_map"]
        if isinstance(pm_dict, dict):
            for loss_name, loss_params in pm_dict.items():
                if not hasattr(pm_losses, loss_name) or not isinstance(loss_params, dict):
                    continue
                kwargs = loss_params.copy()
                weight = kwargs.pop("weight", 1.0)
                if weight == 0.0:
                    continue
                pm_loss_fns.append((getattr(pm_losses, loss_name), weight, kwargs))

    # ── 解析 shell_weighting ──────────────────────────────────────────────────
    dc_cfg          = loss_cfg.get("data_consistency", {})
    shell_weighting = dc_cfg.get("shell_weighting", "none") if isinstance(dc_cfg, dict) else "none"
    shell_weights_tensor = _build_shell_weights(shell_weighting, dc_cfg, volume)

    dc_loss_fns = []
    MSE_exist = False
    SSIM_exist = False
    if "data_consistency" in loss_cfg and loss_cfg["data_consistency"]:
        dc_dict = loss_cfg["data_consistency"]
        if isinstance(dc_dict, dict):
            for loss_name, loss_params in dc_dict.items():
                if hasattr(dc_losses, loss_name) and isinstance(loss_params, dict):
                    kwargs = loss_params.copy()
                    weight = kwargs.pop("weight", 1.0)
                    if weight == 0.0 and loss_name != "MSE":
                        continue
                    # SSIM_loss：build 阶段注入 data_range=volume.signal_max；
                    # voxelwise/coords 无空间维度，直接跳过，无需在 step_fn 中运行时判断
                    if loss_name == "SSIM":
                        if cfg["data"].get("dataset", "voxelwise") in ("voxelwise", "3Dcoord", "coords"):
                            continue
                        kwargs["data_range"] = volume.signal_max
                        SSIM_exist = True
                    if loss_name == "Rician":
                        kwargs["s_noise"] = _build_s_noise(
                            cfg,
                            kwargs.get("s_noise", "auto"),
                            volume,
                            device,
                        )
                    if shell_weights_tensor is not None:
                        kwargs["shell_weights"] = shell_weights_tensor
                    dc_loss_fns.append((getattr(dc_losses, loss_name), weight, kwargs))
                    if loss_name == "MSE":
                        MSE_exist = True
    if not MSE_exist:
        dc_loss_fns.append((dc_losses.MSE, 0.0, {}))

    bval_cpu = bval.detach().float().cpu()
    shell_channels = [
        (_shell_label(shell.item()), torch.where(bval_cpu == shell)[0])
        for shell in torch.unique(bval_cpu, sorted=True)
    ]
    model_input_key = _model_input_key(cfg)

    def step_fn(model, batch, device):
        batch_d    = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                      for k, v in batch.items()}
        signal     = batch_d["signal"].float()
        batch_d["signal"] = signal

        # ── 前向 ────────────────────────────────────────────────────────────
        mask   = batch_d.get("mask")
        numel = signal.numel()
        if model_input_key not in batch_d:
            raise KeyError(
                f"Model '{cfg.get('model', {}).get('name')}' expects batch['{model_input_key}'], "
                f"but batch only has keys: {list(batch_d.keys())}"
            )
        model_input = batch_d[model_input_key]
        if model_input_key == "coords" and pm_loss_fns:
            needs_coord_grad = any(
                loss_fn.__name__ in ("CoordGradL2", "CoordBending")
                for loss_fn, _, _ in pm_loss_fns
            )
            if needs_coord_grad:
                model_input = model_input.detach().clone().requires_grad_(True)
        params = model(model_input, mask=mask) if mask is not None else model(model_input)

        # ── 展平到脑内体素（physics forward 在展平体素上做）───────────────────
        signal_flat, mask_flat = volume.flatten_batch(batch_d)
        params_in    = volume.flatten_spatial(params, mask_flat)

        total_loss = 0.0
        loss_all = {}

        if use_kl and hasattr(model, "kl_loss"):
            kl_loss = model.kl_loss * kl_weight * kl_beta
            total_loss = total_loss + kl_loss
            loss_all["KL"] = kl_loss.item()

        # ── 1. 网络输出空间 Parameter Map 损失（如 TV）──────────────────
        if pm_loss_fns:
            spatial_mask = batch_d.get("mask")
            if spatial_mask is not None and spatial_mask.ndim == params.ndim - 1:
                spatial_mask = spatial_mask.unsqueeze(1)

            params_spatial_eval = params

            for loss_fn, weight, kwargs in pm_loss_fns:
                if loss_fn.__name__ == "TV" and params_spatial_eval.ndim < 3:
                    continue
                if loss_fn.__name__ in ("CoordGradL2", "CoordBending"):
                    pm_loss = loss_fn(params_spatial_eval, coords=model_input, mask=spatial_mask, **kwargs)
                elif loss_fn.__name__ == "CoordFiniteDiffL2":
                    pm_loss = loss_fn(
                        params_spatial_eval,
                        coords=model_input,
                        model=model,
                        mask=spatial_mask,
                        **kwargs,
                    )
                else:
                    pm_loss = loss_fn(params_spatial_eval, mask=spatial_mask, **kwargs)
                loss_all[loss_fn.__name__] = pm_loss.item() * weight
                total_loss = total_loss + pm_loss * weight

        # ── 2. 展平脑内体素后转成物理参数 ───────────────────────────────────
        if require_S0:
            b0_mask = torch.abs(bval.to(device=signal_flat.device)) < 25
            S0_flat = signal_flat[:, b0_mask].mean(dim=1)
            physics_params = physics_instance.output_to_param(
                params_in, S0=S0_flat
            )
        else:
            physics_params = physics_instance.output_to_param(params_in)

        # ── 3. 前向物理模型模拟 Signal 后计算 data consistency ──────────────────────
        signal_pred_flat = physics_instance.forward(physics_params)   # (M, N) 或 (B, N)

        if mask_flat is not None:
            B_   = signal.shape[0]
            N_   = signal_pred_flat.shape[1]
            spat = signal.shape[2:]

            full = torch.zeros(B_ * mask_flat.numel() // B_,
                               N_, device=signal.device, dtype=signal.dtype)
            full[mask_flat] = signal_pred_flat
            perm_back = [0, -1] + list(range(1, 1 + len(spat)))
            signal_pred = full.reshape(B_, *spat, N_).permute(*perm_back).contiguous()
        else:
            signal_pred = signal_pred_flat

        for loss_fn, weight, kwargs in dc_loss_fns:
            dc_loss_val = loss_fn(numel, signal, signal_pred, mask=mask, **kwargs)
            if loss_fn.__name__ == 'MSE':
                loss_all['MSE_unweighted'] = dc_loss_val.item()
            loss_all[loss_fn.__name__] = dc_loss_val.item() * weight
            total_loss = total_loss + dc_loss_val * weight

        with torch.no_grad():
            for shell_label, channel_idx in shell_channels:
                channel_idx = channel_idx.to(signal.device)
                shell_signal = signal.index_select(1, channel_idx)
                shell_pred = signal_pred.index_select(1, channel_idx)
                loss_all[f"MSE_b{shell_label}"] = dc_losses.MSE_metric(shell_signal, shell_pred, mask=mask).item()
                if SSIM_exist:
                    loss_all[f"SSIM_b{shell_label}"] = dc_losses.SSIM_metric(
                        shell_signal,
                        shell_pred,
                        mask=mask,
                        data_range=volume.signal_max,
                    ).item()

        loss_all["total_loss"] = total_loss.item()

        return total_loss, loss_all
    return step_fn
