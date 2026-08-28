import torch
import nibabel as nib
from glob import glob
import os
import numpy as np

import library.dataio.preprocess as preprocess_module
from library.dataio.denoise import apply_denoise
from library.utility.qtlib import *

# ---------------------------------------------------------------------------
# 内部工具：加载 diffusion MRI 原始文件
# ---------------------------------------------------------------------------

def _load_diffusion_data(dpDiff, fpDenoised=None):
    """
    从目录读取 DWI 体积、bval、bvec、mask，返回基础 tensor。

    Returns
    -------
    diff   : (H, W, Z, N_grad)  float32 tensor
    bval   : (N_grad,)           float32 tensor
    bvec   : (3, N_grad)         float32 tensor
    mask   : (H, W, Z)           bool ndarray
    affine : (4, 4)              float64 ndarray
    """
    if fpDenoised is not None:
        fpDiff = fpDenoised
    else:
        fpDiff = glob(os.path.join(dpDiff, '*_diff.nii.gz'))[0]
    fpBval = glob(os.path.join(dpDiff, '*_diff.bval'))[0]
    fpBvec = glob(os.path.join(dpDiff, '*_diff.bvec'))[0]
    fpMask = glob(os.path.join(dpDiff, '*_weight_mask.nii.gz'))[0]

    nii    = nib.load(fpDiff)
    diff   = torch.from_numpy(nii.get_fdata()).float()              # (H,W,Z,N)
    affine = nii.affine                                             # (4,4)
    bval   = torch.from_numpy(np.loadtxt(fpBval)).float().view(-1) # (N,)
    bvec   = torch.from_numpy(np.loadtxt(fpBvec)).float().T        # (3,N)
    if bvec.shape[0] != 3:
        bvec = bvec.T
    mask   = nib.load(fpMask).get_fdata().astype(bool)             # (H,W,Z)

    H, W, Z, N = diff.shape
    assert N == bval.shape[0] == bvec.shape[1], "dimension mismatch"
    assert (H, W, Z) == mask.shape,              "mask shape mismatch"

    return diff, bval, bvec, mask, affine


def _compute_bbox(mask: np.ndarray):
    """计算 bool mask 的紧致包围盒，返回 (h0,h1, w0,w1, d0,d1)。"""
    coords = np.argwhere(mask)
    h0, w0, d0 = coords.min(axis=0)
    h1, w1, d1 = coords.max(axis=0) + 1
    return h0, h1, w0, w1, d0, d1


def _apply_preprocess(diff, mask, bval, preprocess_fn):
    """
    只对 mask 内体素归一化，返回 (归一化后的体积, norm_factor)。
    diff : (H, W, Z, N) float32 tensor（in-place 不修改原 tensor）
    """
    H, W, Z, N = diff.shape
    mask_flat   = mask.reshape(-1)                          # (H*W*Z,) bool
    flat        = diff.reshape(-1, N)                       # (H*W*Z, N)
    brain_norm, norm_factor = preprocess_fn(flat[mask_flat].float(), bval)
    flat_out            = torch.zeros_like(flat)
    flat_out[mask_flat] = brain_norm
    return flat_out.reshape(H, W, Z, N), norm_factor


# ---------------------------------------------------------------------------
# 基类
# ---------------------------------------------------------------------------

class DiffusionDataset(torch.utils.data.Dataset):
    """
    Diffusion MRI 数据集基类。

    封装所有子类共用的逻辑：
      - 文件加载、预处理、bbox 裁剪
      - _volume / _mask tensor 构建
      - GPU preload（volume + mask）
      - orig_mask 属性
      - train/val 分割工具 _make_split

    子类需实现：
      _setup(data_cfg, seed, split)  — 完成子类特化初始化，设置 train_ids / val_ids
      __len__, __getitem__, sample_batch
    """

    def __init__(self, cfg: dict, device=None, preload_gpu: bool = False,
                 seed: int = 42, split: bool = True):
        super().__init__()

        data_cfg        = cfg["data"]
        dpDiff          = data_cfg["data_dir"]
        preprocess_name = data_cfg.get("preprocess", "normalize_b0_mean")

        if not hasattr(preprocess_module, preprocess_name):
            raise ValueError(f"Unknown preprocess function: '{preprocess_name}'")
        preprocess_fn = getattr(preprocess_module, preprocess_name)

        # ── 去噪 ──────────────────────────────────────────────────────────
        denoise_cfg  = data_cfg.get("denoise", None)
        denoise_name = denoise_cfg.get("name") if isinstance(denoise_cfg, dict) else None
        denoise_disabled = denoise_name is None or str(denoise_name).strip().lower() == "none"
        if not denoise_disabled:
            denoise_dict = denoise_cfg.get(denoise_name, {})
            fpDiff = glob(os.path.join(dpDiff, '*_diff.nii.gz'))[0]
            fpMask = glob(os.path.join(dpDiff, '*_diff_mask.nii.gz'))[0]
            dpDenoise = os.path.join(dpDiff, "denoised")
            fpDenoised = apply_denoise(fpDiff, dpDenoise, fpMask, denoise_name, **denoise_dict)
        else:
            fpDenoised = None

        # ── 加载 ──────────────────────────────────────────────────────────
        diff, bval, bvec, mask, affine = _load_diffusion_data(dpDiff, fpDenoised)
        H, W, Z, N = diff.shape

        # ── 预处理 ────────────────────────────────────────────────────────
        diff, norm_factor = _apply_preprocess(diff, mask, bval, preprocess_fn)
        self.signal_max = diff.max().item()  # 供外部使用，计算SSIM loss的时候使用这个值

        # ── 包围盒裁剪 ────────────────────────────────────────────────────
        h0, h1, w0, w1, d0, d1 = _compute_bbox(mask)
        diff_crop = diff[h0:h1, w0:w1, d0:d1, :]    # (H', W', D', N)
        mask_crop = mask[h0:h1, w0:w1, d0:d1]        # (H', W', D') bool
        # ── 公共属性 ───────────────────────────────────────────────────────
        self._volume      = diff_crop.float()                                # (H', W', D', N)
        self._mask        = torch.from_numpy(mask_crop.astype(np.float32))  # (H', W', D')
        self._mask_crop_np = mask_crop                                       # bool ndarray，供 orig_mask 用
        self.bval         = bval
        self.bvec         = bvec
        self.norm_factor  = norm_factor
        self.affine       = affine
        self.device       = device
        self._orig_shape  = (H, W, Z)
        self._bbox        = (h0, h1, w0, w1, d0, d1)
        # ── 每个方向所属 shell 的平均信号（预处理后，脑内体素均值）──────────
        # shell_mean_signal: (N,)，同一 shell 内所有方向共享同一均值
        mask_flat_np = mask_crop.reshape(-1)                                 # (H'*W'*D',) bool
        flat_vol     = diff_crop.reshape(-1, N)                              # (H'*W'*D', N)
        brain_mean   = flat_vol[mask_flat_np].mean(dim=0)                    # (N,) 每个方向的脑内均值
        shells       = bval.unique(sorted=True)                              # unique b-values
        shell_mean   = torch.zeros(N)
        for s in shells:
            idx = (bval == s)
            shell_mean[idx] = brain_mean[idx].mean()
        self.shell_mean_signal = shell_mean                                  # (N,)

        # ── 可选：preload 到 GPU ───────────────────────────────────────────
        self._preloaded = preload_gpu and device is not None
        if self._preloaded:
            self._volume = self._volume.to(device)
            self._mask   = self._mask.to(device)
        # ── 子类特化初始化 ─────────────────────────────────────────────────
        self._setup(data_cfg, H, W, Z, N, seed, split)

    # ── 子类必须实现 ───────────────────────────────────────────────────────

    def _setup(self, data_cfg, H, W, Z, N, seed, split):
        """完成子类特化初始化（构建索引、设置 train_ids / val_ids 等）。"""
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError

    def __getitem__(self, idx):
        raise NotImplementedError

    def sample_batch(self, ids) -> dict:
        raise NotImplementedError

    # ── 公共工具 ───────────────────────────────────────────────────────────

    def _make_split(self, n: int, val_split: float, seed: int):
        """随机划分 n 个索引，返回 (train_ids, val_ids)。"""
        rng   = np.random.default_rng(seed)
        ids   = rng.permutation(n)
        n_val = max(1, int(n * val_split))
        return ids[n_val:].tolist(), ids[:n_val].tolist()

    def flatten_batch(self, batch: dict):
        """
        将 sample_batch 返回的任意维度 batch 展平为脑内体素级别。

        支持
        ----
        voxelwise  : signal (B, N)               → signal_in (B, N)，无 mask
        slicewise  : signal (B, N, H, W)         → signal_in (M, N)，M = mask 内体素数
        patchwise  : signal (B, N, P, P, P)      → signal_in (M, N)

        返回
        ----
        signal_in : (M, N) float Tensor，仅脑内体素
        mask_flat : (B*spatial,) bool Tensor | None（voxelwise 时为 None）
        """
        signal = batch["signal"]
        if signal.dim() == 2:                      # voxelwise
            return signal, None
        mask_flat = batch["mask"].reshape(-1).bool()
        signal_in = self.flatten_spatial(signal, mask_flat)
        return signal_in, mask_flat

    @staticmethod
    def flatten_spatial(x: torch.Tensor, mask_flat) -> torch.Tensor:
        """
        将 (B, C, *spatial) tensor 按 mask 展平为 (M, C)。

        mask_flat : (B * prod(spatial),) bool
        返回      : (M, C)，M = mask_flat.sum()

        voxelwise（mask_flat=None）时直接返回 x，不做任何操作。
        """
        if mask_flat is None:
            return x
        C    = x.shape[1]
        spat = x.shape[2:]                              # () / (H,W) / (P,P,P)
        # (B, C, *spat) → (B, *spat, C) → (B*prod(spat), C) → mask → (M, C)
        n_spat = len(spat)
        perm   = [0] + list(range(2, 2 + n_spat)) + [1]
        return x.permute(*perm).reshape(-1, C)[mask_flat]

    @property
    def orig_mask(self) -> np.ndarray:
        """原始空间 (H,W,Z) bool mask，由 bbox + 裁剪 mask 重建，不额外占用内存。"""
        h0, h1, w0, w1, d0, d1 = self._bbox
        orig = np.zeros(self._orig_shape, dtype=bool)
        orig[h0:h1, w0:w1, d0:d1] = self._mask_crop_np
        return orig


# ---------------------------------------------------------------------------
# 子类
# ---------------------------------------------------------------------------

class voxelwise(DiffusionDataset):
    """
    逐体素数据集，用于 MLP / 隐式神经表示。

    Attributes
    ----------
    train_ids / val_ids : mask 内体素行索引列表
    sample_batch(ids)   : {"signal": (B, N_grad)}
    bval, bvec, norm_factor, affine : 供外部使用
    """

    def _setup(self, data_cfg, H, W, Z, N, seed, split):
        # 展平为脑内体素（_volume 可能已在 GPU 上）
        mask_flat   = self._mask_crop_np.reshape(-1)
        self._brain = self._volume.reshape(-1, N)[mask_flat]  # (N_voxel, N_grad)
        n = self._brain.shape[0]

        print(f"voxelwise: {(H,W,Z)} → bbox {self._mask_crop_np.shape}")

        if split:
            self.train_ids, self.val_ids = self._make_split(
                n, data_cfg.get("val_split", 0.1), seed)
            print(f"voxelwise: {len(self.train_ids)} train / {len(self.val_ids)} val voxels")
        else:
            self.train_ids = list(range(n))
            self.val_ids   = list(range(n))
            print(f"voxelwise: {n} voxels (no split)")

    def __len__(self):
        return self._brain.shape[0]

    def __getitem__(self, idx):
        return {"signal": self._brain[idx]}

    def sample_batch(self, ids) -> dict:
        """返回 {"signal": (B, N_grad)}。"""
        idx    = torch.tensor(ids, device=self._brain.device)
        signal = self._brain[idx]
        if not self._preloaded and self.device is not None:
            signal = signal.to(self.device)
        return {"signal": signal}


class coords(DiffusionDataset):
    """
    3D 坐标数据集，用于隐式神经表示。

    网络输入为裁剪后 bbox 内的归一化坐标，监督信号仍为对应 DWI。

    Attributes
    ----------
    train_ids / val_ids : mask 内体素行索引列表
    sample_batch(ids)   : {"coords": (B, coord_dim), "signal": (B, N_grad)}
    """

    name = "coords"
    coord_dim = 3

    def _setup(self, data_cfg, H, W, Z, N, seed, split):
        mask_flat = self._mask_crop_np.reshape(-1)
        grid = self._make_coord_grid(data_cfg)

        self._brain = self._volume.reshape(-1, N)[mask_flat]
        self._coords = grid.reshape(-1, self.coord_dim)[mask_flat]
        n = self._brain.shape[0]

        print(f"{self.name}: {(H,W,Z)} → bbox {self._mask_crop_np.shape}")

        if split:
            self.train_ids, self.val_ids = self._make_split(
                n, data_cfg.get("val_split", 0.1), seed)
            print(f"{self.name}: {len(self.train_ids)} train / {len(self.val_ids)} val voxels")
        else:
            self.train_ids = list(range(n))
            self.val_ids   = list(range(n))
            print(f"{self.name}: {n} voxels (no split)")

    def _normalized_axes(self):
        device = self._volume.device
        return [
            torch.linspace(0.0, 1.0, steps=s, device=device, dtype=self._volume.dtype)
            if s > 1 else torch.zeros(1, device=device, dtype=self._volume.dtype)
            for s in self._mask_crop_np.shape
        ]

    def _make_coord_grid(self, data_cfg):
        axes = self._normalized_axes()
        return torch.stack(torch.meshgrid(*axes, indexing="ij"), dim=-1)

    def __len__(self):
        return self._brain.shape[0]

    def __getitem__(self, idx):
        return {"coords": self._coords[idx], "signal": self._brain[idx]}

    def sample_batch(self, ids) -> dict:
        """返回 {"coords": (B, 3), "signal": (B, N_grad)}。"""
        idx    = torch.tensor(ids, device=self._brain.device)
        coords = self._coords[idx]
        signal = self._brain[idx]
        if not self._preloaded and self.device is not None:
            coords = coords.to(self.device)
            signal = signal.to(self.device)
        return {"coords": coords, "signal": signal}


class slicewise(DiffusionDataset):
    """
    以单方向切片为单位的数据集，用于 2D 卷积网络整层输入训练/推理。

    orientation : "axial"    → 沿 Z 轴切，每片 (N_grad, H', W')
                  "coronal"  → 沿 Y 轴切，每片 (N_grad, H', D')
                  "sagittal" → 沿 X 轴切，每片 (N_grad, W', D')

    Attributes
    ----------
    train_ids / val_ids : 有效切片的轴向索引列表（相对于裁剪后体积）
    sample_batch(ids)   : {"signal": (B, N_grad, H', W'), "mask": (B, H', W')}
    bval, bvec, norm_factor, affine : 供外部使用
    """

    _ORIENT_AXIS = {"axial": 2, "coronal": 1, "sagittal": 0}

    def _setup(self, data_cfg, H, W, Z, N, seed, split):
        orientation = data_cfg.get("orientation", "axial")
        if orientation not in self._ORIENT_AXIS:
            raise ValueError(f"orientation must be one of {list(self._ORIENT_AXIS)}, got '{orientation}'")

        self.orientation = orientation
        self._axis       = self._ORIENT_AXIS[orientation]

        print(f"slicewise: {(H,W,Z)} → bbox {tuple(self._volume.shape[:3])}")

        # 有效切片索引（至少有一个脑内体素）
        D_crop    = self._volume.shape[self._axis]
        all_ids   = [i for i in range(D_crop) if self._get_mask_slice(i).any()]
        self._all_ids = all_ids

        if split:
            rng   = np.random.default_rng(seed)
            ids   = np.array(all_ids)
            rng.shuffle(ids)
            n_val = max(1, int(len(ids) * data_cfg.get("val_split", 0.1)))
            self.val_ids   = ids[:n_val].tolist()
            self.train_ids = ids[n_val:].tolist()
            print(f"slicewise: {len(self.train_ids)} train / {len(self.val_ids)} val slices")
        else:
            self.train_ids = []
            self.val_ids   = all_ids
            print(f"slicewise: {len(all_ids)} slices (no split)")

    # ── 内部工具 ──────────────────────────────────────────────────────────

    def _get_mask_slice(self, i):
        ax = self._axis
        if ax == 0: return self._mask[i, :, :]
        if ax == 1: return self._mask[:, i, :]
        return self._mask[:, :, i]

    def _get_slice(self, i):
        ax = self._axis
        if ax == 0:
            sig = self._volume[i, :, :, :]
            msk = self._mask[i, :, :]
        elif ax == 1:
            sig = self._volume[:, i, :, :]
            msk = self._mask[:, i, :]
        else:
            sig = self._volume[:, :, i, :]
            msk = self._mask[:, :, i]
        return sig.permute(2, 0, 1), msk      # (N, *, *), (*, *)

    # ── DataLoader 兼容接口 ────────────────────────────────────────────────

    def __len__(self):
        return len(self._all_ids)

    def __getitem__(self, idx):
        sig, msk = self._get_slice(self._all_ids[idx])
        return {"signal": sig, "mask": msk}

    def sample_batch(self, ids) -> dict:
        """返回 {"signal": (B, N_grad, H', W'), "mask": (B, H', W')}。"""
        idx = torch.as_tensor(ids, device=self._volume.device)
        ax  = self._axis
        if ax == 0:
            signal = self._volume[idx, :, :, :].permute(0, 3, 1, 2).contiguous()
            mask   = self._mask[idx, :, :]
        elif ax == 1:
            signal = self._volume[:, idx, :, :].permute(1, 3, 0, 2).contiguous()
            mask   = self._mask[:, idx, :].permute(1, 0, 2).contiguous()
        else:
            signal = self._volume[:, :, idx, :].permute(2, 3, 0, 1).contiguous()
            mask   = self._mask[:, :, idx].permute(2, 0, 1).contiguous()
        if not self._preloaded and self.device is not None:
            signal = signal.to(self.device)
            mask   = mask.to(self.device)
        return {"signal": signal, "mask": mask}


class patchwise(DiffusionDataset):
    """
    以 3D patch 为单位的数据集，用于 3D 卷积网络训练/推理。
    Patch 在脑内 bbox 上均匀平铺（可重叠），由 block_ind 计算。

    data_cfg 相关字段
    -----------------
    patch_size : int   patch 边长（立方体），默认 64；若 mask bbox 任一轴更小，会自动降到可用尺寸
    sz_pad     : int   block_ind 的 sz_pad 参数，默认 0

    Attributes
    ----------
    train_ids / val_ids : patch 索引列表
    sample_batch(ids)   : {"signal": (B, N_grad, P, P, P), "mask": (B, P, P, P)}
    bval, bvec, norm_factor, affine : 供外部使用
    ind_block : (N_patch, 6) int ndarray，每行 [x0, x1, y0, y1, z0, z1]（包含端点）
    """

    def _setup(self, data_cfg, H, W, Z, N, seed, split):
        requested_patch_size = int(data_cfg.get("patch_size", 64))
        sz_pad     = data_cfg.get("sz_pad", 0)

        print(f"patchwise: {(H,W,Z)} → bbox {tuple(self._volume.shape[:3])}")

        max_patch_size = int(min(self._mask_crop_np.shape))
        if max_patch_size < 1:
            raise ValueError("patchwise requires a non-empty mask bbox")
        patch_size = min(requested_patch_size, max_patch_size)
        if patch_size != requested_patch_size:
            print(
                "patchwise: requested patch_size="
                f"{requested_patch_size} exceeds mask bbox {tuple(self._mask_crop_np.shape)}; "
                f"using patch_size={patch_size}"
            )

        self.patch_size = patch_size

        ind_block, _ = block_ind(self._mask_crop_np, sz_block=patch_size, sz_pad=sz_pad)
        self.ind_block = ind_block
        n = ind_block.shape[0]

        print(f"patchwise: {n} patches (size={patch_size}, sz_pad={sz_pad})")

        if split:
            self.train_ids, self.val_ids = self._make_split(
                n, data_cfg.get("val_split", 0.1), seed)
            print(f"patchwise: {len(self.train_ids)} train / {len(self.val_ids)} val patches")
        else:
            self.train_ids = list(range(n))
            self.val_ids   = list(range(n))
            print(f"patchwise: {n} patches (no split)")

    # ── DataLoader 兼容接口 ────────────────────────────────────────────────

    def __len__(self):
        return self.ind_block.shape[0]

    def __getitem__(self, idx):
        sig, msk = self._get_patch(idx)
        return {"signal": sig, "mask": msk}

    def sample_batch(self, ids) -> dict:
        """返回 {"signal": (B, N_grad, P, P, P), "mask": (B, P, P, P)}。"""
        signal = extract_block_torch_vec(self._volume, self.ind_block[ids])  # (B,P,P,P,N)
        signal = signal.permute(0, 4, 1, 2, 3).contiguous()                 # (B,N,P,P,P)
        mask   = extract_block_torch_vec(
            self._mask.unsqueeze(-1), self.ind_block[ids]
        ).squeeze(-1)                                                         # (B,P,P,P)
        if not self._preloaded and self.device is not None:
            signal = signal.to(self.device)
            mask   = mask.to(self.device)
        return {"signal": signal, "mask": mask}


# ---------------------------------------------------------------------------
# 工厂函数
# ---------------------------------------------------------------------------

def build_dataset(cfg: dict, device=None, preload_gpu: bool = False,
                  seed: int = 42, split: bool = True) -> DiffusionDataset:
    """
    根据 config['data']['dataset'] 构建对应数据集。

    支持: "voxelwise" | "coords" | "slicewise" | "patchwise"
    """
    _REGISTRY = {
        "voxelwise": voxelwise,
        "coords": coords,
        "slicewise": slicewise,
        "patchwise": patchwise,
    }
    data_cfg = cfg["data"]
    kind = data_cfg.get("dataset", "voxelwise")
    if kind not in _REGISTRY:
        raise ValueError(f"Unknown dataset type: '{kind}'，可选: {list(_REGISTRY)}")
    return _REGISTRY[kind](cfg, device=device, preload_gpu=preload_gpu,
                           seed=seed, split=split)
