import torch
import torch.nn.functional as F
from torchmetrics.image.ssim import StructuralSimilarityIndexMeasure as _SSIM_metric


def _apply_mask(loss_map, mask):
    for _ in range(loss_map.dim() - mask.dim()):
        mask = mask.unsqueeze(1)
    return loss_map * mask


def _apply_shell_weights(loss_map, shell_weights):
    shape = [1] * loss_map.dim()
    shape[1] = -1
    return loss_map * shell_weights.view(*shape)


def _channel_view(value, ref):
    if not isinstance(value, torch.Tensor) or value.dim() != 1:
        return value
    if value.numel() != ref.shape[1]:
        return value
    shape = [1] * ref.dim()
    shape[1] = -1
    return value.view(*shape)


def _dc_loss(numel, s, pred, loss_type, 
             mask, s_noise, shell_weights, beta):
    """
    统一数据一致性损失计算核心。

    loss_type        : "MSE" | "MAE" | "Huber" | "rician"
    """

    if loss_type == "MSE":
        loss = (s - pred) ** 2
    elif loss_type == "MAE":
        loss = torch.abs(s - pred)
    elif loss_type == "Huber":  # Huber
        loss = F.smooth_l1_loss(pred, s, reduction='none', beta=beta)
    elif loss_type == "rician":
        if s_noise is None:
            raise ValueError("Rician loss requires s_noise")
        sigma = _channel_view(s_noise.to(device=s.device, dtype=s.dtype), s).clamp_min(1e-8)
        s_pos = s.clamp_min(1e-8)
        pred_pos = pred.clamp_min(0.0)
        sigma2 = sigma ** 2
        z = s_pos * pred_pos / sigma2
        log_pdf = (
            torch.log(s_pos / sigma2)
            - (s_pos ** 2 + pred_pos ** 2) / (2 * sigma2)
            + torch.log(torch.special.i0e(z).clamp_min(1e-12)) + z
        )
        loss = -log_pdf
        z_ref = s_pos * s_pos / sigma2
        ref_log_pdf = (
            torch.log(s_pos / sigma2)
            - (s_pos ** 2 + s_pos ** 2) / (2 * sigma2)
            + torch.log(torch.special.i0e(z_ref).clamp_min(1e-12)) + z_ref
        )
        loss = (loss + ref_log_pdf) * (2 * sigma2)

    if shell_weights is not None:
        loss = _apply_shell_weights(loss, shell_weights)
    if mask is not None:
        loss = _apply_mask(loss, mask)
    return loss.sum() / numel


def MSE(numel, signal, signal_pred, mask=None, s_noise=None, shell_weights=None,):
    """MSE 损失"""
    return _dc_loss(numel, signal, signal_pred, "MSE", 
                    mask, s_noise, shell_weights, beta=1.0)


def MAE(numel, signal, signal_pred, mask=None, s_noise=None, shell_weights=None,):
    """MAE 损失"""
    return _dc_loss(numel, signal, signal_pred, "MAE", 
                    mask, s_noise, shell_weights, beta=1.0)


def Huber(numel, signal, signal_pred, mask=None, s_noise=None, shell_weights=None, beta=1.0):
    """Huber 损失"""
    return _dc_loss(numel, signal, signal_pred, "Huber", 
                    mask, s_noise, shell_weights, beta=beta)

def Rician(numel, signal, signal_pred, mask=None, s_noise=None,
           shell_weights=None, beta=1.0, scaled=True):
    """Rician 损失"""
    return _dc_loss(numel, signal, signal_pred, "rician", 
                    mask, s_noise, shell_weights, beta=beta)


def MSE_metric(signal, signal_pred, mask=None):
    loss = (signal - signal_pred) ** 2
    if mask is None:
        return loss.mean()

    m = mask.float()
    for _ in range(loss.dim() - mask.dim()):
        m = m.unsqueeze(1)
    return (loss * m).sum() / (m.sum() * loss.shape[1]).clamp_min(1.0)


def SSIM_metric(signal, signal_pred, mask=None, data_range=1.0):
    if mask is not None:
        m = mask.clone().float()
        for _ in range(signal.dim() - mask.dim()):
            m = m.unsqueeze(1)
        signal = signal * m
        signal_pred = signal_pred * m

    B, N = signal.shape[:2]
    spatial = signal.shape[2:]

    x = signal.reshape(B * N, 1, *spatial)
    y = signal_pred.reshape(B * N, 1, *spatial)

    ssim_fn = _SSIM_metric(data_range=float(data_range), reduction='none').to(x.device)
    return ssim_fn(y, x).reshape(B, N).mean()


def SSIM(numel, signal, signal_pred, mask=None, shell_weights=None, data_range=1.0):
    """
    结构相似性损失，返回 1 - mean_SSIM。

    signal / signal_pred : (B, N, H, W)    ← slicewise
                           (B, N, H, W, D) ← patchwise
    mask                 : (B, H, W) 或 (B, H, W, D)，或 None
    shell_weights        : (N,) 或 None
    data_range           : 信号动态范围，由 build_blocks.py 注入 volume.signal_max

    voxelwise 数据集无空间维度，不应调用本函数（build_blocks.py 在调用侧跳过）。
    """
    if mask is not None:
        m = mask.clone().float()
        for _ in range(signal.dim() - mask.dim()):
            m = m.unsqueeze(1)
        signal      = signal      * m
        signal_pred = signal_pred * m

    B, N    = signal.shape[:2]
    spatial = signal.shape[2:]

    x = signal.reshape(B * N, 1, *spatial)
    y = signal_pred.reshape(B * N, 1, *spatial)

    ssim_fn     = _SSIM_metric(data_range=float(data_range), reduction='none').to(x.device)
    per_channel = ssim_fn(y, x).reshape(B, N)   # (B, N)

    if shell_weights is not None:
        ssim_val = (per_channel * shell_weights.unsqueeze(0)).mean()
    else:
        ssim_val = per_channel.mean()

    return 1.0 - ssim_val
