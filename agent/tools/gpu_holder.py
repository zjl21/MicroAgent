import threading
import subprocess
import time

import torch

DEFAULT_PROCESS_MIB = 40000
GPU_SLOT_UNAVAILABLE = "GPU_SLOT_UNAVAILABLE"


def parse_gpu_ids(value):
    if value is None or str(value).strip() == "":
        return None
    return [int(item.strip()) for item in str(value).split(",") if item.strip()]


def query_gpu_memory(gpu_ids=None):
    """Return {gpu_id: (free_mib, total_mib)} without initializing CUDA."""
    try:
        result = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=index,memory.free,memory.total",
                "--format=csv,nounits,noheader",
            ],
            encoding="utf-8",
        )
        wanted = set(gpu_ids) if gpu_ids is not None else None
        memory = {}
        for line in result.splitlines():
            if not line.strip():
                continue
            gpu_id, free_mib, total_mib = [int(part.strip()) for part in line.split(",")]
            if wanted is None or gpu_id in wanted:
                memory[gpu_id] = (free_mib, total_mib)
        return memory
    except Exception:
        n_gpus = torch.cuda.device_count()
        ids = gpu_ids if gpu_ids is not None else range(n_gpus)
        memory = {}
        for gpu_id in ids:
            free, total = torch.cuda.mem_get_info(gpu_id)
            memory[int(gpu_id)] = (free // (1024 * 1024), total // (1024 * 1024))
        return memory


class GPUResourceManager:
    """Thread-safe logical GPU slot allocator.

    The manager tracks processes started by this agent in memory and uses
    nvidia-smi free memory to stay away from other users' processes.
    """

    def __init__(
        self,
        process_mib=DEFAULT_PROCESS_MIB,
        max_processes=8,
        poll_sec=1800,
        gpu_ids=None,
        quiet=False,
        slot_tolerance_mib=0,
    ):
        self.process_mib = int(process_mib)
        self.max_processes = int(max_processes)
        self.poll_sec = int(poll_sec)
        self.gpu_ids = gpu_ids or sorted(query_gpu_memory().keys())
        self.quiet = quiet
        self.slot_tolerance_mib = int(slot_tolerance_mib)
        self.active_counts = {gpu_id: 0 for gpu_id in self.gpu_ids}
        self.blocked_until = {gpu_id: 0.0 for gpu_id in self.gpu_ids}
        self.cond = threading.Condition()

    def _active_processes(self):
        return sum(self.active_counts.values())

    def _available_assignments(self):
        if self._active_processes() >= self.max_processes:
            return []

        memory = query_gpu_memory(self.gpu_ids)
        candidates = []
        now = time.time()
        for gpu_id in self.gpu_ids:
            if self.blocked_until.get(gpu_id, 0.0) > now:
                continue
            free_mib, total_mib = memory.get(gpu_id, (0, 0))
            owned_slots = self.active_counts.get(gpu_id, 0)
            card_slots = max(1, (total_mib + self.slot_tolerance_mib) // self.process_mib)
            logical_room = card_slots - owned_slots
            if logical_room < 1 or free_mib + self.slot_tolerance_mib < self.process_mib:
                continue
            # Pack slots onto cards already used by this agent before opening a
            # new card. Ties use ascending GPU id for deterministic low-to-high
            # allocation.
            candidates.append((-owned_slots, gpu_id, -free_mib, free_mib, total_mib))

        candidates.sort()
        return candidates

    def has_available_slot(self):
        with self.cond:
            return bool(self._available_assignments())

    def has_required_free_memory(self, gpu_id):
        memory = query_gpu_memory([gpu_id])
        free_mib, _ = memory.get(gpu_id, (0, 0))
        return free_mib + self.slot_tolerance_mib >= self.process_mib

    def acquire(self, label="task"):
        warned = False
        with self.cond:
            while True:
                candidates = self._available_assignments()
                if candidates:
                    neg_owned_slots, gpu_id, neg_free_mib, _, total_mib = candidates[0]
                    owned_slots = -neg_owned_slots
                    self.active_counts[gpu_id] += 1
                    if not self.quiet:
                        print(
                            f"{label}: GPU {gpu_id}, active={self._active_processes()}/{self.max_processes}, "
                            f"free={-neg_free_mib}MiB, used_slots={owned_slots + 1}"
                        )
                    return {
                        "gpu_id": gpu_id,
                        "process_mib": self.process_mib,
                    }

                if not warned and not self.quiet:
                    print(f"{label}: waiting for GPU slot ({self.process_mib}MiB)")
                    warned = True
                self.cond.wait(timeout=self.poll_sec)

    def release(self, gpu_id):
        with self.cond:
            if gpu_id in self.active_counts and self.active_counts[gpu_id] > 0:
                self.active_counts[gpu_id] -= 1
            self.cond.notify_all()

    def mark_unavailable(self, gpu_id, cooldown_sec=None):
        cooldown_sec = self.poll_sec if cooldown_sec is None else int(cooldown_sec)
        with self.cond:
            if gpu_id in self.blocked_until:
                self.blocked_until[gpu_id] = time.time() + cooldown_sec
            self.cond.notify_all()


class GPUHolder:
    """GPU memory helper.

    - setup_process_memory(): call inside train/infer process to cap PyTorch usage
      and reserve that process's own cache.
    - occupy_on_device(): optional external occupancy from the agent process.
    """

    _cuda_lock = threading.Lock()

    def __init__(self, reserve_mib=1024, device_id=None, quiet=False, hold_mib=None):
        self.reserve_mib = int(reserve_mib)
        self.hold_mib = int(hold_mib) if hold_mib is not None else None
        self.device_id = int(device_id) if device_id is not None else self._select_device()
        self.device = f"cuda:{self.device_id}"
        self.quiet = quiet
        self.tensor = None
        self.occupied_mib = 0

    def _select_device(self):
        memory = query_gpu_memory()
        if not memory:
            return 0
        return max(memory.items(), key=lambda item: item[1][0])[0]

    def occupy_on_device(self):
        """Externally occupy this task's slot in the agent process."""
        if self.tensor is not None:
            return True

        free_mib, _ = query_gpu_memory([self.device_id]).get(self.device_id, (0, 0))
        if self.hold_mib is None:
            occupy_mib = max(0, free_mib - self.reserve_mib)
        else:
            occupy_mib = min(self.hold_mib, max(0, free_mib - self.reserve_mib))
            if occupy_mib + self.reserve_mib < self.hold_mib:
                return False

        while occupy_mib > 0:
            try:
                with self._cuda_lock, torch.cuda.device(self.device_id):
                    self.tensor = torch.empty(occupy_mib * 1024 * 1024, dtype=torch.uint8, device=self.device)
                    self.tensor.fill_(0)
                self.occupied_mib = occupy_mib
                if not self.quiet:
                    print(f"GPU holder occupied {occupy_mib}MiB on {self.device}")
                return True
            except torch.cuda.OutOfMemoryError:
                occupy_mib -= 512
                self.tensor = None
                with self._cuda_lock, torch.cuda.device(self.device_id):
                    torch.cuda.empty_cache()
        return False

    def release(self):
        if self.tensor is not None:
            with self._cuda_lock, torch.cuda.device(self.device_id):
                del self.tensor
                self.tensor = None
                self.occupied_mib = 0
                torch.cuda.empty_cache()

    @staticmethod
    def setup_process_memory(
        device,
        limit_mib=None,
        reserve_cache=True,
        guard_mib=1024,
        required=False,
        tolerance_mib=2048,
    ):
        """Limit and pre-reserve memory inside the current PyTorch process."""
        if device.type != "cuda" or limit_mib is None:
            return

        free, total = torch.cuda.mem_get_info(device)
        free_mib = free // (1024 * 1024)
        total_mib = total // (1024 * 1024)
        limit_mib = min(int(limit_mib), int(total_mib - guard_mib))
        if limit_mib <= 0:
            return

        if required and free_mib + int(tolerance_mib) < limit_mib:
            raise RuntimeError(
                f"{GPU_SLOT_UNAVAILABLE}: need {limit_mib}MiB on {device}, free {free_mib}MiB"
            )

        torch.cuda.set_per_process_memory_fraction(limit_mib / total_mib, device)

        if reserve_cache:
            reserve_mib = max(0, min(limit_mib - guard_mib, free_mib - guard_mib))
            if reserve_mib <= 0:
                if required:
                    raise RuntimeError(
                        f"{GPU_SLOT_UNAVAILABLE}: cannot reserve cache on {device}, free {free_mib}MiB"
                    )
                return
            try:
                tensor = torch.empty(reserve_mib * 1024 * 1024, dtype=torch.uint8, device=device)
                tensor.fill_(0)
                del tensor
                # Do not empty_cache(): keep this process's allocator cache reserved
                # so later training tensors can reuse it and external processes cannot.
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                if required:
                    raise RuntimeError(
                        f"{GPU_SLOT_UNAVAILABLE}: failed to reserve {reserve_mib}MiB on {device}"
                    )
