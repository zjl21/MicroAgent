import subprocess
import threading
import torch
from typing import Callable


def _get_free_mib(device_id: int) -> int:
    try:
        result = subprocess.check_output(
            ['nvidia-smi', f'--id={device_id}',
             '--query-gpu=memory.free', '--format=csv,nounits,noheader'],
            encoding='utf-8')
        return int(result.strip())
    except Exception:
        free, _ = torch.cuda.mem_get_info(device_id)
        return free // (1024 * 1024)


def _get_used_mib(device_id: int) -> int:
    try:
        result = subprocess.check_output(
            ['nvidia-smi', f'--id={device_id}',
             '--query-gpu=memory.used', '--format=csv,nounits,noheader'],
            encoding='utf-8')
        return int(result.strip())
    except Exception:
        _, total = torch.cuda.mem_get_info(device_id)
        free, _ = torch.cuda.mem_get_info(device_id)
        return (total - free) // (1024 * 1024)


def build_gpu_schedule(probe_fn: Callable[[str], None], reserve_mib: int = 512, max_jobs: int = 12) -> list[str]:
    """
    自适应分配 GPU 并发槽，返回 device 字符串列表（可直接轮询）。

    Parameters
    ----------
    probe_fn : Callable[[str], None]
        接受一个 device 字符串（如 'cuda:1'），完整跑一次目标任务。
        用于实测单次峰值显存。
    reserve_mib : int
        每张卡保留的安全余量（MiB），默认 512。
    max_jobs : int
        并发上限，0 表示不限制（由显存决定）。建议设为任务总数。

    Returns
    -------
    device_slots : list[str]
        展开的 device 列表，如 ['cuda:1','cuda:1','cuda:2','cuda:2',...]。
        len(device_slots) 即建议的 n_jobs。

    Example
    -------
    >>> device_slots = build_gpu_schedule(
    ...     lambda dev: denoise_2D_slice(slice_2d, mask_2d, device=dev, tsince=tsince)
    ... )
    >>> results = Parallel(n_jobs=len(device_slots))(
    ...     delayed(my_fn)(data[i], device=device_slots[i % len(device_slots)])
    ...     for i in range(N)
    ... )
    """
    n_gpus = torch.cuda.device_count()

    # 找最空的卡做 probe
    probe_gpu = max(range(n_gpus), key=_get_free_mib)
    probe_device = f'cuda:{probe_gpu}'

    torch.cuda.init()

    # 后台线程轮询 nvidia-smi，抓 probe 运行期间的真实峰值
    peak_mib = [0]
    stop_flag = threading.Event()

    def _poll():
        while not stop_flag.is_set():
            peak_mib[0] = max(peak_mib[0], _get_used_mib(probe_gpu))
            stop_flag.wait(timeout=0.1)

    poller = threading.Thread(target=_poll, daemon=True)
    poller.start()
    probe_fn(probe_device)
    stop_flag.set()
    poller.join()
    per_task_mib = int(peak_mib[0] * 1.1)  # 1.1x 安全系数，避免多进程竞争时 OOM
    torch.cuda.empty_cache()

    # 重新查询（probe 跑完后显存已释放）
    free_list = sorted([(i, _get_free_mib(i)) for i in range(n_gpus)], key=lambda x: -x[1])

    schedule = []
    for gpu_id, free_mib in free_list:
        slots = max(0, free_mib - reserve_mib) // per_task_mib
        if slots > 0:
            schedule.append((f'cuda:{gpu_id}', int(slots)))

    if not schedule:
        detail = ' | '.join(f'cuda:{i}: {m}MiB free' for i, m in free_list)
        raise RuntimeError(
            f'[gpu_strategy] 所有 GPU 空闲显存均不足以运行一个任务（每任务需 {per_task_mib}MiB，保留余量 {reserve_mib}MiB）\n'
            f'  各卡状态: {detail}'
        )

    device_slots = [d for d, s in schedule for _ in range(s)]
    if max_jobs > 0 and len(device_slots) > max_jobs:
        device_slots = device_slots[:max_jobs]

    return device_slots
