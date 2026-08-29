
import torch
import torch.nn.functional as F

def _apply_mask(loss_map, mask):
    for _ in range(loss_map.dim() - mask.dim()):
        mask = mask.unsqueeze(1)
    return loss_map * mask

def TV(params_map, mask=None):
    """
    N维总变分损失 (Total Variation, TV Loss)。可以自适应 1D、2D 或 3D 空间结构。
    作用于生成的**物理参数图** (而不是 signal)，用于促使生成的参数图空间上更加平滑、保留边缘结构，去除孤立的噪声点。
    
    params_map: (B, C, ...) 空间的参数图，例如 (B, C, W), (B, C, H, W), (B, C, D, H, W) 等。
    mask:       (B, 1, ...) 或其他形状兼容的指定有效脑区掩码。
    """
    if mask is not None:
        params_map = _apply_mask(params_map, mask)
    # 用每个channel的最大值归一化
    params_map = params_map / (params_map.max(dim=1, keepdim=True)[0] + 1e-8)
    tv_loss = 0.0
    # 默认前两维为 Batch 和 Channels，从第 2 维开始计算剩余每个空间维度上的相邻间距差异
    for spatial_dim in range(2, params_map.ndim):
        tv = torch.abs(torch.diff(params_map, dim=spatial_dim))
        tv_loss += tv.sum()
        
    return tv_loss/params_map.numel()


def constrain(params_map, mask=None, physics_name='DKI'):
    """
    约束损失 (Constraint Loss)，用于约束生成的物理参数图在合理范围内，避免出现过大或过小的异常值。例如通过约束使 DKI kurtosis “正定”
    """
    if mask is not None:
        mask_bool = (mask > 0)
        if mask_bool.ndim == params_map.ndim and mask_bool.shape[1] == 1:
            mask_bool = mask_bool.squeeze(1)
        elif mask_bool.ndim != params_map.ndim - 1:
            raise ValueError(
                f"mask shape {tuple(mask.shape)} is not compatible with params_map shape {tuple(params_map.shape)}"
            )

        # Keep channel dimension, flatten only batch+spatial dims selected by mask.
        params_map = params_map.movedim(1, -1)[mask_bool]
    try:
        physics_module   = __import__(f"library.physics.{physics_name}", fromlist=[physics_name])
        constrain_loss = physics_module.constrain(params_map)
    except:
        return torch.tensor(0.0, device=params_map.device)
        
    return constrain_loss


def _channel_weights(params_map, weights=None):
    if weights is None or weights == []:
        return None
    w = torch.as_tensor(weights, device=params_map.device, dtype=params_map.dtype)
    if w.numel() != params_map.shape[-1]:
        raise ValueError(
            f"weights length {w.numel()} does not match params channel count {params_map.shape[-1]}"
        )
    return w.view(*([1] * (params_map.ndim - 1)), -1)


def CoordGradL2(params_map, coords=None, mask=None, weights=None, normalize=True):
    """
    坐标网络的一阶平滑项：惩罚 d(params)/d(coords) 的 L2 范数。

    params_map : (B, C) for 3Dcoord
    coords     : (B, 3), requires_grad=True

    相比 TV，它不依赖规则 grid 邻接关系，适合 coordinate batch 随机采样训练。
    """
    if coords is None:
        raise ValueError("CoordGradL2 requires coords tensor.")
    if not coords.requires_grad:
        raise ValueError("CoordGradL2 requires coords.requires_grad=True.")
    if params_map.ndim != 2:
        return torch.tensor(0.0, device=params_map.device, dtype=params_map.dtype)

    params_eval = params_map
    if normalize:
        scale = params_eval.detach().abs().amax(dim=0, keepdim=True).clamp_min(1e-6)
        params_eval = params_eval / scale

    w = _channel_weights(params_eval, weights)
    losses = []
    for c in range(params_eval.shape[1]):
        grad_c = torch.autograd.grad(
            params_eval[:, c].sum(),
            coords,
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]
        val = grad_c.pow(2).sum(dim=-1)
        if w is not None:
            val = val * w[..., c]
        losses.append(val)
    return torch.stack(losses, dim=-1).mean()


def CoordBending(params_map, coords=None, mask=None, weights=None, normalize=True):
    """
    坐标网络的二阶 bending energy：惩罚二阶导数。

    这个正则更强、更贵；对某些 CUDA hash-grid 实现，二阶梯度可能不如
    Fourier/SIREN-style encoder 稳定。建议先用 CoordGradL2。
    """
    if coords is None:
        raise ValueError("CoordBending requires coords tensor.")
    if not coords.requires_grad:
        raise ValueError("CoordBending requires coords.requires_grad=True.")
    if params_map.ndim != 2:
        return torch.tensor(0.0, device=params_map.device, dtype=params_map.dtype)

    params_eval = params_map
    if normalize:
        scale = params_eval.detach().abs().amax(dim=0, keepdim=True).clamp_min(1e-6)
        params_eval = params_eval / scale

    w = _channel_weights(params_eval, weights)
    losses = []
    dim = coords.shape[1]
    for c in range(params_eval.shape[1]):
        grad_c = torch.autograd.grad(
            params_eval[:, c].sum(),
            coords,
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]
        second_terms = []
        for j in range(dim):
            second_j = torch.autograd.grad(
                grad_c[:, j].sum(),
                coords,
                create_graph=True,
                retain_graph=True,
                only_inputs=True,
            )[0]
            second_terms.append(second_j.pow(2).sum(dim=-1))
        val = torch.stack(second_terms, dim=-1).sum(dim=-1)
        if w is not None:
            val = val * w[..., c]
        losses.append(val)
    return torch.stack(losses, dim=-1).mean()


def CoordFiniteDiffL2(params_map, coords=None, model=None, mask=None,
                      eps=1e-3, weights=None, normalize=True):
    """
    坐标网络的有限差分平滑项：惩罚邻近坐标输出变化。

    这个版本不需要对 coords 做 autograd，因此兼容 tiny-cuda-nn HashGrid。
    代价是每个空间维度额外做 2 次模型前向。
    """
    if coords is None:
        raise ValueError("CoordFiniteDiffL2 requires coords tensor.")
    if model is None:
        raise ValueError("CoordFiniteDiffL2 requires model.")
    if params_map.ndim != 2:
        return torch.tensor(0.0, device=params_map.device, dtype=params_map.dtype)

    coords_eval = coords.detach()
    params_ref = params_map
    if normalize:
        scale = params_ref.detach().abs().amax(dim=0, keepdim=True).clamp_min(1e-6)
    else:
        scale = 1.0

    w = _channel_weights(params_map, weights)
    losses = []
    dim = coords_eval.shape[1]
    eps = float(eps)
    for j in range(dim):
        delta = torch.zeros_like(coords_eval)
        delta[:, j] = eps
        plus = (coords_eval + delta).clamp(0.0, 1.0)
        minus = (coords_eval - delta).clamp(0.0, 1.0)
        p_plus = model(plus).float()
        p_minus = model(minus).float()
        grad_fd = (p_plus - p_minus) / (2.0 * eps)
        if normalize:
            grad_fd = grad_fd / scale
        val = grad_fd.pow(2)
        if w is not None:
            val = val * w
        losses.append(val.mean())
    return torch.stack(losses).mean()
