import numpy as np


def _hanning_qt(n: int, ratio: float = 1.5) -> np.ndarray:
    k = np.arange(1, n + 1, dtype=np.float64) - (n / 2 + 0.5)
    return 0.5 * (1 + np.cos(np.pi * k / n * ratio))


def _center_pad_4d(vols: np.ndarray, pad_shape: tuple[int, int, int]) -> np.ndarray:
    spatial = vols.shape[:3]
    if any(p < s for p, s in zip(pad_shape, spatial)):
        raise ValueError(f"pad_shape {pad_shape} must be >= input spatial shape {spatial}.")
    out = np.zeros((*pad_shape, vols.shape[3]), dtype=vols.dtype)
    start = [(p - s) // 2 for p, s in zip(pad_shape, spatial)]
    out[
        start[0]:start[0] + spatial[0],
        start[1]:start[1] + spatial[1],
        start[2]:start[2] + spatial[2],
        :,
    ] = vols
    return out


def _center_crop_4d(vols: np.ndarray, output_shape: tuple[int, int, int]) -> np.ndarray:
    spatial = vols.shape[:3]
    if any(o > s for o, s in zip(output_shape, spatial)):
        raise ValueError(f"output_shape {output_shape} must be <= input spatial shape {spatial}.")
    start = [(s - o) // 2 for s, o in zip(spatial, output_shape)]
    return vols[
        start[0]:start[0] + output_shape[0],
        start[1]:start[1] + output_shape[1],
        start[2]:start[2] + output_shape[2],
        :,
    ]


def kspace_downsample(
    vols: np.ndarray,
    source_voxel_size,
    target_voxel_size,
    pad_shape=None,
    output_shape=None,
) -> np.ndarray:
    """
    Fourier-domain downsample matching kspace_dsp.m.

    vols: 3D or 4D array with spatial axes first. target_voxel_size must be
    greater than or equal to source_voxel_size for every spatial axis.
    """
    vols = np.asarray(vols)
    input_was_3d = vols.ndim == 3
    if input_was_3d:
        vols = vols[..., np.newaxis]
    if vols.ndim != 4:
        raise ValueError(f"vols must be 3D or 4D, got shape {vols.shape}.")

    source_voxel_size = np.broadcast_to(np.asarray(source_voxel_size, dtype=float), (3,))
    target_voxel_size = np.broadcast_to(np.asarray(target_voxel_size, dtype=float), (3,))
    dsp_ratio = source_voxel_size / target_voxel_size
    if np.any(dsp_ratio <= 0) or np.any(dsp_ratio > 1):
        raise ValueError(f"target_voxel_size must be >= source_voxel_size, got ratio {dsp_ratio}.")

    if output_shape is not None:
        output_shape = tuple(int(v) for v in output_shape)

    work = vols.astype(np.float32, copy=False)
    if pad_shape is not None:
        work = _center_pad_4d(work, tuple(int(v) for v in pad_shape))

    spatial = work.shape[:3]
    ker_3d = np.ones(spatial, dtype=np.float32)
    for axis, ratio in enumerate(dsp_ratio):
        n = spatial[axis]
        if np.isclose(ratio, 1.0):
            ker = np.ones(n, dtype=np.float32)
        else:
            n1_float = n * ratio
            n1 = int(round(n1_float + ((n % 2) - (n1_float % 2))))
            n1 = min(max(1, n1), n)
            filter_han = _hanning_qt(n)
            filter1_han = _hanning_qt(n1)
            filter1_han_pad = np.zeros(n, dtype=np.float64)
            half_sz = (n - n1) // 2
            filter1_han_pad[half_sz:half_sz + n1] = filter1_han
            ker = (filter1_han_pad / filter_han).astype(np.float32)

        shape = [1, 1, 1]
        shape[axis] = n
        ker_3d *= ker.reshape(shape)

    coords = np.argwhere(ker_3d > 0)
    mins = coords.min(axis=0)
    maxs = coords.max(axis=0) + 1
    ker_crop = ker_3d[mins[0]:maxs[0], mins[1]:maxs[1], mins[2]:maxs[2]]
    out = np.zeros((*ker_crop.shape, work.shape[3]), dtype=np.float32)

    for i in range(work.shape[3]):
        spectrum = np.fft.fftshift(np.fft.fftn(work[..., i]))
        spectrum_crop = spectrum[mins[0]:maxs[0], mins[1]:maxs[1], mins[2]:maxs[2]] * ker_crop
        out[..., i] = np.real(np.fft.ifftn(np.fft.ifftshift(spectrum_crop))).astype(np.float32)

    if output_shape is not None and output_shape != out.shape[:3]:
        out = _center_crop_4d(out, output_shape)
    return out[..., 0] if input_was_3d else out


def fill_nan_with_neighbors(tensor, kernel_size=3, max_iter=10):
    """
    用周围 3D 邻居体素的非 NaN 均值替换数组中的 NaN 值。
    
    参数:
    tensor (np.ndarray): 输入数组，形状支持 3D (D, H, W), 4D (C, D, H, W) 或 5D (B, C, D, H, W)
    kernel_size (int): 3D 邻域窗口大小，默认 3 (即 3x3x3=27 个体素，中心为自身，26个邻居)
    max_iter (int): 最大迭代次数，处理连续大片 NaN 时从边缘逐渐向内修复。
    """
    assert kernel_size % 2 == 1, "卷积核大小必须是奇数 (如 3, 5, 7)"
    array = np.asarray(tensor)
    if not np.issubdtype(array.dtype, np.floating):
        array = array.astype(np.float32)

    orig_dim = array.ndim

    if orig_dim == 3:
        array = array[np.newaxis, np.newaxis, ...]
    elif orig_dim == 4:
        array = array[np.newaxis, ...]
    elif orig_dim != 5:
        raise ValueError("输入数组必须是 3D, 4D 或 5D")

    result = array.copy()
    center = kernel_size // 2

    for i in range(max_iter):
        nan_mask = np.isnan(result)
        if not nan_mask.any():
            print(f"迭代 {i} 次后已无 NaN。")
            break

        valid_mask = (~nan_mask).astype(result.dtype, copy=False)
        zeroed = np.where(nan_mask, 0, result)

        padded_data = np.pad(
            zeroed,
            ((0, 0), (0, 0), (center, center), (center, center), (center, center)),
            mode="constant",
            constant_values=0,
        )
        padded_mask = np.pad(
            valid_mask,
            ((0, 0), (0, 0), (center, center), (center, center), (center, center)),
            mode="constant",
            constant_values=0,
        )

        neighbor_sum = np.zeros_like(result, dtype=result.dtype)
        neighbor_count = np.zeros_like(result, dtype=result.dtype)

        for dz in range(kernel_size):
            for dy in range(kernel_size):
                for dx in range(kernel_size):
                    if dz == center and dy == center and dx == center:
                        continue
                    neighbor_sum += padded_data[:, :, dz:dz + result.shape[2], dy:dy + result.shape[3], dx:dx + result.shape[4]]
                    neighbor_count += padded_mask[:, :, dz:dz + result.shape[2], dy:dy + result.shape[3], dx:dx + result.shape[4]]

        neighbor_avg = neighbor_sum / np.maximum(neighbor_count, 1e-8)
        update_mask = nan_mask & (neighbor_count > 0)
        result = np.where(update_mask, neighbor_avg, result)

    if orig_dim == 3:
        result = result.squeeze(0).squeeze(0)
    elif orig_dim == 4:
        result = result.squeeze(0)

    return result
