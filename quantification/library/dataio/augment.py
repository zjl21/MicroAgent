"""
augment.py — slicewise / patchwise 数据增强

每种增强以独立的 50% 概率触发，在 config 中通过 data.augment 列表选择：

    data:
      augment:
        - flip_x                    # 左右翻转（沿最后一维）
        - flip_y                    # 上下翻转（沿倒数第二维）
        - flip_z                    # 深度翻转（沿倒数第三维，仅 patchwise 有效）
        - rot180                    # 180° 旋转（= flip_x + flip_y，不改变空间尺寸）
        - noise                     # 对信号加 Gaussian 噪声，默认 noise_std_ratio=0.05
        # 或带参数写法：
        - name: noise
          noise_std_ratio: 0.1

slicewise : signal (N,H,W) / (B,N,H,W)，mask (H,W) / (B,H,W)
patchwise : signal (N,P,P,P) / (B,N,P,P,P)，mask (P,P,P) / (B,P,P,P)
"""

import torch


# ---------------------------------------------------------------------------
# 单个增强操作
# ---------------------------------------------------------------------------

def _flip_x(signal, mask, **_):
    return signal.flip(2), mask.flip(1)


def _flip_y(signal, mask, **_):
    return signal.flip(3), mask.flip(2)


def _flip_z(signal, mask, **_):
    """深度翻转：沿倒数第三维，patchwise 专用。"""
    return signal.flip(4), mask.flip(3)

def _rot90(signal, mask, **_):
    """90度旋转：仅 patchwise 有效，等价于先转置再 flip_y。"""
    return signal.transpose(2, 3).flip(3), mask.transpose(1, 2).flip(2)


def _rot180(signal, mask, **_):
    return signal.flip(2).flip(3), mask.flip(1).flip(2)


def _noise(signal, mask, std_ratio=0.05, **_):
    # 应该是b0的std，而非signal整体的std，后者可能过大导致噪声过强
    b0_std = signal[:,0,...].std()
    return signal + torch.randn_like(signal) * (b0_std * std_ratio), mask


_AUG_FN = {
    "flip_x": _flip_x,
    "flip_y": _flip_y,
    "flip_z": _flip_z,
    "rot90": _rot90,
    "rot180": _rot180,
    "noise":  _noise,
}

AVAILABLE = list(_AUG_FN.keys())


# ---------------------------------------------------------------------------
# 数据增强
# ---------------------------------------------------------------------------

def apply_augmentations(batch: dict, aug_list: list) -> dict:
    """
    对已在 GPU 上的 mini-batch 做 on-the-fly 增强。

    slicewise : signal (B, N_grad, H, W)，mask (B, H, W)
    patchwise : signal (B, N_grad, P, P, P)，mask (B, P, P, P)

    aug_list 每项可以是字符串或带参数的字典：
        - "flip_x"
        - {"name": "noise", "p": 0.3, "std_ratio": 0.05}

    p : 每个样本被施加该增强的概率，默认 0.5。
    """
    if not aug_list:
        return batch
    signal = batch["signal"]
    if signal.dim() not in (4, 5):
        return batch
    mask = batch["mask"]
    B = signal.shape[0]

    for item in aug_list:
        if isinstance(item, dict):
            name = item["name"]
            p = float(item.get("p", 0.5))
            item_kwargs = {k: v for k, v in item.items() if k not in ("name", "p")}
        else:
            name = item
            p = 0.5
            item_kwargs = {}

        if ( (name == "flip_z") or (name == "rot90") ) and signal.dim() == 4:
            continue                                # slicewise 无深度维，跳过
        flip = torch.rand(B, device=signal.device) < p
        if not flip.any():
            continue
        signal[flip], mask[flip] = _AUG_FN[name](signal[flip], mask[flip], **item_kwargs)
    return {"signal": signal, "mask": mask}
