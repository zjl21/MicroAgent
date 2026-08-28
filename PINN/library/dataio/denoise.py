import json
import os
import fcntl
from glob import glob


def _read_cache(cache_path):
    if not os.path.exists(cache_path):
        return {}
    with open(cache_path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _write_cache_atomic(cache_path, cache):
    temporary = f"{cache_path}.tmp-{os.getpid()}"
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(cache, handle, indent=2, ensure_ascii=False)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, cache_path)


def apply_denoise(fpDiff, dpDenoise, fpMask, method, **kwargs):
    """
    对单个数据进行去噪，结果写入 dpDenoise/<method>_<idx>.nii.gz。

    参数
    ----
    fpDiff    : str  输入 NIfTI 文件路径
    dpDenoise : str  输出目录
    method    : str  去噪方法，可选 "MPPCA" / "BM4D"；
                     设为 "none" 时直接返回原始 DWI
    fpMask    : str  脑掩膜文件路径
    kwargs    : 传递给具体去噪方法 run() 的参数，会覆盖同名默认值

    缓存机制
    --------
    在 dpDenoise 下维护 <method>.json，结构为：
        {
            "0": {"patchradius": 1, "searchradius": 3, ...},
            "1": {"p": 0.3, "lr": 1e-4, ...},
            ...
        }
    idx 按记录顺序自动递增。若本次 run_kwargs 与某条记录完全一致，
    则直接复用 <method>_<idx>.nii.gz，不重复运行去噪。

    返回
    ----
    str : 去噪结果的完整文件路径
    """
    if method is None or str(method).strip().lower() == "none":
        return fpDiff

    default_kwargs = {
        "MPPCA": {},
        "BM4D":  {},
    }

    try:
        mod = __import__(f"library.dataio.denoise_lib.{method}", fromlist=[method])
        cls = getattr(mod, method)
    except (ImportError, AttributeError):
        raise ValueError(
            f"未知去噪方法: {method!r}，可选: {list(default_kwargs.keys())}"
        )

    # 合并默认参数与用户传入参数（用户传入优先）
    run_kwargs = {**default_kwargs.get(method, {}), **kwargs}

    # ── 缓存读取 ──────────────────────────────────────────────────
    os.makedirs(dpDenoise, exist_ok=True)
    cache_path = os.path.join(dpDenoise, f"{method}.json")

    # Multiple Random workers may request the same denoiser with different
    # parameters. Hold a per-method file lock across cache allocation and the
    # actual denoising run so two workers cannot claim the same numeric index
    # or overwrite the JSON cache. The cache itself is committed atomically.
    lock_path = f"{cache_path}.lock"
    with open(lock_path, "a+", encoding="utf-8") as lock_handle:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
        cache = _read_cache(cache_path)

        for idx, params in cache.items():
            if params == run_kwargs:
                output_file = os.path.join(dpDenoise, f"{method}_{idx}.nii.gz")
                if os.path.exists(output_file):
                    return output_file

        numeric_indices = [int(idx) for idx in cache if str(idx).isdigit()]
        new_idx = str(max(numeric_indices, default=-1) + 1)
        output_file = os.path.join(dpDenoise, f"{method}_{new_idx}.nii.gz")
        temporary_output = os.path.join(
            dpDenoise,
            f"{method}_{new_idx}.tmp-{os.getpid()}.nii.gz",
        )

        denoiser = cls(
            input_file=fpDiff,
            output_file=temporary_output,
            mask_file=fpMask,
        )
        try:
            denoiser.run(**run_kwargs)
            os.replace(temporary_output, output_file)
        finally:
            if os.path.exists(temporary_output):
                os.remove(temporary_output)

        cache[new_idx] = run_kwargs
        _write_cache_atomic(cache_path, cache)
        return output_file


def find_denoise(dpDenoise, method, **kwargs):
    cache_path = os.path.join(dpDenoise, f"{method}.json")

    default_kwargs = {
        "MPPCA": {},
        "BM4D":  {},
    }
    # 合并默认参数与用户传入参数（用户传入优先）
    run_kwargs = {**default_kwargs.get(method, {}), **kwargs}

    cache = _read_cache(cache_path)

    # 在已有记录中查找是否有完全一致的 params
    hit_idx = None
    for idx, params in cache.items():
        if params == run_kwargs:
            hit_idx = idx
            break

    output_file = os.path.join(dpDenoise, f"{method}_{hit_idx}.nii.gz")
    return output_file
