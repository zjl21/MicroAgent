"""
trainer.py — 通用训练器

使用方式::

    from library.trainer import Trainer
    from toolbox.utility.iolib import load_config

    def step_fn(model, batch, device):
        # 用户自定义：数据搬运 + 前向 + 损失计算
        x = batch["signal"].to(device)
        loss = criterion(model(x), ...)
        return loss

    trainer = Trainer(model, train_loader, val_loader, step_fn, cfg)
    trainer.train()
"""

import os
import math
import json
import time
import random
from datetime import datetime
import torch
from torch.optim import Adam, AdamW, SGD
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR, ReduceLROnPlateau, ExponentialLR, LRScheduler



# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class Trainer:
    """
    通用训练器。与模型结构、数据集、损失函数完全解耦。

    Parameters
    ----------
    model   : nn.Module
    volume  : GPUVolumeVoxelwise | GPUVolumeSlicewise
        已预加载到 GPU 的体积采样器，需提供：
        - volume.train_ids / val_ids  : 训练/验证索引列表
        - volume.sample_batch(ids)    : 返回 {signal, [mask]} GPU batch
        - volume.device               : torch.device
    step_fn : callable(model, batch, device) -> scalar Tensor
    cfg     : dict  从 config.yaml 加载的配置字典
    """

    def __init__(self, model, volume, step_fn, cfg: dict, cfg_path: str = ""):
        self.model   = model
        self.volume  = volume
        self.step_fn = step_fn
        self.cfg     = cfg
        self.device  = volume.device

        env_cfg   = cfg.get("env", {})
        train_cfg = cfg.get("training", {})
        sched_cfg = train_cfg.get("scheduler", {})

        base_dir = os.path.dirname(os.path.abspath(cfg_path))
        default_exp_dir = base_dir
        default_ckpt_dir = os.path.join(base_dir, "checkpoints")

        self.ckpt_dir = env_cfg.get("ckpt_dir", default_ckpt_dir)
        os.makedirs(self.ckpt_dir, exist_ok=True)

        # ── 训练超参数 ────────────────────────────────────────────────────
        self.epochs     = train_cfg.get("epochs", 100)
        self.grad_clip  = train_cfg.get("grad_clip", None)
        self.log_every  = train_cfg.get("log_every", 1)
        self.ckpt_every = max(1, int(train_cfg.get("ckpt_every", 50)))
        self.resume_cfg = train_cfg.get("resume", {})
        if isinstance(self.resume_cfg, bool):
            self.resume_cfg = {"enabled": self.resume_cfg}
        elif not isinstance(self.resume_cfg, dict):
            self.resume_cfg = {}
        # Keep training.resume in config for compatibility, but always enable
        # checkpoint resume at runtime.
        self.resume_enabled = True
        self.resume_strict = bool(self.resume_cfg.get("strict", True))
        self.resume_path = self.resume_cfg.get("path", "")
        self.start_epoch = 1
        self.resumed_from = ""

        # ── 优化器 ────────────────────────────────────────────────────────
        self.model.to(self.device)
        self.optimizer = self._build_optimizer(train_cfg)

        # ── 调度器 ────────────────────────────────────────────────────────
        self.scheduler = self._build_scheduler(sched_cfg)

        self.best_val_loss = float("inf")
        self.best_epoch = 0

        # ── 早停 ──────────────────────────────────────────────────────────
        es_cfg = train_cfg.get("early_stopping", {})
        self.es_enabled   = bool(es_cfg.get("enabled", False))
        self.es_patience  = int(es_cfg.get("patience", 50))
        self.es_min_delta = float(es_cfg.get("min_delta", 1e-6))
        self._es_counter  = 0

        # ── JSON 训练日志 ──────────────────────────────────────────────────
        self.log_path = os.path.join(default_exp_dir, "training_log.json")
        self._log = self._load_or_init_log(cfg_path)

        # ── 数据增强 ───────────────────────────────────────────────────────
        self.aug_list = cfg.get("data", {}).get("augment", [])

        # ── 断点恢复 ───────────────────────────────────────────────────────
        if self.resume_enabled:
            self._resume_if_available()
        self._log["resume"] = {
            "enabled": self.resume_enabled,
            "resumed": bool(self.resumed_from),
            "checkpoint": self.resumed_from,
            "start_epoch": self.start_epoch,
        }
        self._save_log()

    # -----------------------------------------------------------------------
    # 公开接口
    # -----------------------------------------------------------------------

    def train(self):
        t_start = time.time()
        stopped_early = False
        last_epoch = self.start_epoch - 1

        if self.start_epoch > self.epochs:
            print(
                f"Checkpoint already reached epoch {self.start_epoch - 1}; "
                f"configured epochs={self.epochs}. Nothing to train."
            )
            self._finalize_log(last_epoch, stopped_early, 0.0)
            return

        for epoch in range(self.start_epoch, self.epochs + 1):
            last_epoch = epoch
            train_loss, train_comps = self._run_epoch(self.volume.train_ids, train=True)
            val_loss, val_comps   = self._run_epoch(self.volume.val_ids,   train=False)

            if self.scheduler is not None:
                if isinstance(self.scheduler, ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                elif isinstance(self.scheduler, WarmupPlateauLR):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()

            lr = self.optimizer.param_groups[0]["lr"]

            # ── 记录到历史 ──────────────────────────────────────────────
            record = {
                "epoch":      epoch,
                "train_loss": round(train_loss, 8),
                "val_loss":   round(val_loss,   8),
                "train_comps": {k: round(v, 8) for k, v in train_comps.items()},
                "val_comps":   {k: round(v, 8) for k, v in val_comps.items()},
                "lr":         lr,
            }
            self._upsert_history_record(record)
            if epoch % self.log_every == 0:
                self._save_log()

                comp_str = " | "+" ".join([f"{k}: {v:.4f}" for k, v in val_comps.items()])
                print(
                    f"Epoch {epoch}/{self.epochs}"
                    f" | train_loss: {train_loss:.6f}"
                    f" | val_loss: {val_loss:.6f}"
                    f" | lr: {lr:.6e}"
                    f"{comp_str}"
                )

            improved = val_loss < self.best_val_loss - self.es_min_delta
            if improved:
                self.best_val_loss = val_loss
                self.best_epoch = epoch
                self._es_counter = 0
            elif self.es_enabled:
                self._es_counter += 1
                if self._es_counter >= self.es_patience:
                    print(
                        f"Early stopping triggered at epoch {epoch} "
                        f"(no improvement for {self.es_patience} epochs)."
                    )
                    stopped_early = True
                    self._save_ckpt(epoch, val_loss, "latest_model.pth")
                    break

            checkpoint_due = epoch % self.ckpt_every == 0 or epoch == self.epochs
            if checkpoint_due:
                self._save_ckpt(epoch, val_loss, "latest_model.pth")
                if improved:
                    self._save_ckpt(epoch, val_loss, "best_model.pth")

        elapsed = time.time() - t_start
        self._finalize_log(last_epoch, stopped_early, elapsed)
        print("Training finished.")

    # -----------------------------------------------------------------------
    # 内部方法
    # -----------------------------------------------------------------------

    @staticmethod
    def _clip_grad_norm_no_sync(parameters, max_norm: float):
        """
        clip_grad_norm_ 的无 GPU-CPU sync 版本。
        全程在 GPU 上完成，不调用 .item()，不阻塞 GPU 流水线。

        注意：先将 inf/nan 梯度清零，避免 inf×0=NaN 污染权重。
        """
        params = [p for p in parameters if p.grad is not None]
        if not params:
            return
        # inf/nan → 0，防止后续 clip_coef 计算时 inf×0=NaN 污染权重
        for p in params:
            p.grad.nan_to_num_(nan=0.0, posinf=0.0, neginf=0.0)
        norms      = torch.stack([p.grad.detach().norm() for p in params])
        total_norm = norms.norm()                        # 0-dim GPU tensor
        clip_coef  = max_norm / (total_norm + 1e-6)     # 0-dim GPU tensor
        clip_coef  = torch.clamp(clip_coef, max=1.0)    # 不放大，只裁剪
        for p in params:
            p.grad.mul_(clip_coef)

    def _augment(self, batch: dict) -> dict:
        from library.dataio.augment import apply_augmentations
        return apply_augmentations(batch, self.aug_list)

    def _run_epoch(self, ids: list, train: bool) -> tuple:
        self.model.train(train)
        running = torch.zeros(1, device=self.device)
        total   = 0
        batch_size = self.cfg["data"].get("batch_size", 2048)
        running_components = {}

        ids_arr = list(ids)
        if train:
            import random
            random.shuffle(ids_arr)

        ctx = torch.enable_grad() if train else torch.no_grad()
        with ctx:
            for start in range(0, len(ids_arr), batch_size):
                batch = self.volume.sample_batch(ids_arr[start:start + batch_size])
                if train:
                    batch = self._augment(batch)
                
                loss, loss_components = self.step_fn(self.model, batch, self.device)
                
                if train:
                    self.optimizer.zero_grad()
                    loss.backward()
                    if self.grad_clip is not None:
                        self._clip_grad_norm_no_sync(self.model.parameters(), self.grad_clip)
                    self.optimizer.step()
                    
                running += loss.detach()
                for k, v in loss_components.items():
                    if k not in running_components:
                        running_components[k] = torch.zeros(1, device=self.device)
                    running_components[k] += v
                total   += 1

        avg_total = (running / total).item() if total > 0 else float("nan")
        avg_components = {k: (v / total).item() for k, v in running_components.items()} if total > 0 else {}
        return avg_total, avg_components

    # -----------------------------------------------------------------------
    # JSON 日志
    # -----------------------------------------------------------------------

    def _load_or_init_log(self, cfg_path: str) -> dict:
        fresh_log = {
            "config_path": os.path.abspath(cfg_path) if cfg_path else "",
            "training_history": [],
            "summary": {},
        }
        if not self.resume_enabled or not os.path.exists(self.log_path):
            return fresh_log
        try:
            with open(self.log_path, "r") as f:
                log = json.load(f)
            if not isinstance(log.get("training_history"), list):
                log["training_history"] = []
            log.setdefault("summary", {})
            log["config_path"] = fresh_log["config_path"]
            return log
        except (OSError, json.JSONDecodeError) as exc:
            print(f"Could not reuse existing training log: {exc}. Starting a new log.")
            return fresh_log

    def _save_log(self):
        # 先写临时文件，再原子替换，避免 monitor 读到写了一半的 JSON
        tmp_path = self.log_path + ".tmp"
        with open(tmp_path, "w") as f:
            json.dump(self._log, f, indent=2, ensure_ascii=False)
        os.replace(tmp_path, self.log_path)

    def _upsert_history_record(self, record: dict):
        history = self._log.setdefault("training_history", [])
        epoch = int(record.get("epoch", 0))
        for i, old in enumerate(history):
            if int(old.get("epoch", 0)) == epoch:
                history[i] = record
                return
        history.append(record)

    def _finalize_log(self, final_epoch: int, stopped_early: bool, elapsed_sec: float):
        history = self._log["training_history"]
        if not history:
            return

        best_record = min(history, key=lambda r: r["val_loss"])
        last = history[-1]

        self._log["summary"] = {
            "total_epochs_run": final_epoch,
            "stopped_early":    stopped_early,
            "elapsed_sec":      round(elapsed_sec, 1),
            "resumed_from":     self.resumed_from,
            "best_val_loss":    best_record["val_loss"],
            "best_epoch":       best_record["epoch"],
            "final_train_loss": last["train_loss"],
            "final_val_loss":   last["val_loss"],
            "final_train_comps": last.get("train_comps", {}),
            "final_val_comps":  last.get("val_comps", {}),
            "final_lr":         last["lr"],
        }
        self._save_log()

    def _save_ckpt(self, epoch: int, val_loss: float, filename: str):
        state = {
            "epoch":      epoch,
            "val_loss":   val_loss,
            "best_val_loss": self.best_val_loss,
            "best_epoch": self.best_epoch,
            "early_stopping_counter": self._es_counter,
            "config":     self.cfg,
            "saved_at":   datetime.now().isoformat(timespec="seconds"),
            "checkpoint_meta": {
                "epoch": epoch,
                "epochs": self.epochs,
                "lr": self.optimizer.param_groups[0]["lr"],
                "train_size": len(self.volume.train_ids),
                "val_size": len(self.volume.val_ids),
                "parameter_count": sum(p.numel() for p in self.model.parameters()),
                "trainable_parameter_count": sum(p.numel() for p in self.model.parameters() if p.requires_grad),
                "optimizer": self.cfg.get("training", {}).get("optimizer", {}),
                "scheduler": self.cfg.get("training", {}).get("scheduler", {}),
            },
            "model":      (
                self.model.module.state_dict()
                if isinstance(self.model, torch.nn.DataParallel)
                else self.model.state_dict()
            ),
            "optimizer":  self.optimizer.state_dict(),
            "rng_state":  self._get_rng_state(),
        }
        if self.scheduler is not None:
            state["scheduler"] = self.scheduler.state_dict()
        path = os.path.join(self.ckpt_dir, filename)
        tmp_path = path + ".tmp"
        torch.save(state, tmp_path)
        os.replace(tmp_path, path)

    def _resume_if_available(self):
        ckpt_path = self._find_resume_checkpoint()
        if not ckpt_path:
            return
        try:
            ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=False)
        except Exception as exc:
            print(f"Found checkpoint but could not load it: {ckpt_path} ({exc}). Starting from scratch.")
            return

        model_state = ckpt.get("model", ckpt)
        self.model.load_state_dict(model_state, strict=self.resume_strict)

        if "optimizer" in ckpt:
            try:
                self.optimizer.load_state_dict(ckpt["optimizer"])
                self._move_optimizer_state_to_device()
            except ValueError as exc:
                print(f"Could not restore optimizer state from {ckpt_path}: {exc}")
        if self.scheduler is not None and "scheduler" in ckpt:
            try:
                self.scheduler.load_state_dict(ckpt["scheduler"])
            except Exception as exc:
                print(f"Could not restore scheduler state from {ckpt_path}: {exc}")

        self.best_val_loss = float(ckpt.get("best_val_loss", ckpt.get("val_loss", self.best_val_loss)))
        self.best_epoch = int(ckpt.get("best_epoch", ckpt.get("epoch", self.best_epoch)))
        self._es_counter = int(ckpt.get("early_stopping_counter", self._es_counter))
        self.start_epoch = int(ckpt.get("epoch", 0)) + 1
        self.resumed_from = ckpt_path
        if self.resume_cfg.get("restore_rng_state", True):
            self._restore_rng_state(ckpt.get("rng_state", {}))
        self._trim_log_after_epoch(self.start_epoch - 1)
        print(f"Resumed training from {ckpt_path} at epoch {self.start_epoch}.")

    def _find_resume_checkpoint(self) -> str:
        candidates = []
        if self.resume_path:
            candidates.append(self.resume_path)
        candidates.extend([
            os.path.join(self.ckpt_dir, "latest_model.pth"),
            os.path.join(self.ckpt_dir, "best_model.pth"),
        ])
        for path in candidates:
            if path and os.path.exists(path):
                return path
        return ""

    def _trim_log_after_epoch(self, last_finished_epoch: int):
        history = self._log.get("training_history", [])
        self._log["training_history"] = [
            r for r in history if int(r.get("epoch", 0)) <= last_finished_epoch
        ]
        self._log["summary"] = {}

    def _move_optimizer_state_to_device(self):
        for state in self.optimizer.state.values():
            for key, value in state.items():
                if torch.is_tensor(value):
                    state[key] = value.to(self.device)

    def _get_rng_state(self) -> dict:
        state = {
            "python": random.getstate(),
            "torch": torch.get_rng_state(),
        }
        if torch.cuda.is_available():
            state["cuda"] = torch.cuda.get_rng_state_all()
        return state

    def _restore_rng_state(self, state: dict):
        if not state:
            return
        try:
            if "python" in state:
                random.setstate(state["python"])
            if "torch" in state:
                torch.set_rng_state(state["torch"])
            if "cuda" in state and torch.cuda.is_available():
                torch.cuda.set_rng_state_all(state["cuda"])
        except Exception as exc:
            print(f"Could not restore RNG state: {exc}")

    def _build_optimizer(self, train_cfg: dict):
        optimizer_cfg = train_cfg.get('optimizer', 'adam')
        
        # Backward compatibility for flat "optimizer: adam" vs nested "optimizer: {name: adam}"
        if isinstance(optimizer_cfg, str):
            name = optimizer_cfg.lower()
            optimizer_cfg = {"name": name}
        else:
            name = optimizer_cfg.get("name", "adam").lower()
            
        lr            = float(train_cfg.get("lr", 1e-3))
        params        = self.model.parameters()

        # 做一个name到函数的字典映射
        optimizer_map = {
            "adam": Adam,
            "adamw": AdamW,
            "sgd": SGD,
        }
        kawargs = optimizer_cfg[name] if name in optimizer_cfg else {}
        
        # 将科学计数法字符串或可转为 float 的字符串强转为 float
        for k, v in kawargs.items():
            if isinstance(v, str):
                try:
                    kawargs[k] = float(v)
                except ValueError:
                    pass

        if name in optimizer_map:
            return optimizer_map[name](params, lr=lr, **kawargs)
        else:
            raise ValueError(f"Unknown optimizer: {name}")  

    def _build_scheduler(self, sched_cfg: dict):
        stype = sched_cfg.get("name", "none").lower()
        scheduler_map = {
            "none": None,
            "cosine": CosineAnnealingLR,
            "step": StepLR,
            "plateau": ReduceLROnPlateau,
            "exponential": ExponentialLR,
            "cosine_warmup": CosineWarmupLR,
            "cosine_warm_restart": CosineWarmRestartLR,
            "warmup_plateau": WarmupPlateauLR,
        }
        kawargs = sched_cfg[stype] if stype in sched_cfg else {}
        
        # 将科学计数法字符串或可转为 float 的字符串强转为 float
        for k, v in kawargs.items():
            if isinstance(v, str):
                try:
                    kawargs[k] = float(v)
                except ValueError:
                    pass

        if stype in scheduler_map:
            scheduler_cls = scheduler_map[stype]
            if scheduler_cls is None:
                return None
            return scheduler_cls(self.optimizer, **kawargs)
        else:            
            raise ValueError(f"Unknown scheduler type: {stype}")

# ---------------------------------------------------------------------------
# 自定义调度器：带热身的余弦退火
# ---------------------------------------------------------------------------

class CosineWarmupLR(LRScheduler):
    """
    带线性预热的余弦退火学习率调度器。
    
    参数:
        optimizer (Optimizer): 所关联的优化器
        warmup_epochs (int): 热身轮数
        T_max (int): 训练总轮数 (或余弦周期的最大轮数)
        eta_min (float, 可选): 触底时的最小学习率，默认为 0.0
        last_epoch (int, 可选): 最后一个 epoch 的索引，默认为 -1
    """
    def __init__(self, optimizer, warmup_epochs: int, T_max: int, eta_min: float = 0.0, last_epoch: int = -1):
        self.warmup_epochs = int(warmup_epochs)
        self.T_max = int(T_max)
        self.eta_min = float(eta_min)
        super().__init__(optimizer, last_epoch)
        # base_lrs 是由于继承 LRScheduler 自动从 optimizer 中提取出来的初始学习率列表

    def get_lr(self):
        # 1. 如果在热身阶段，做线性爬升
        if self.last_epoch < self.warmup_epochs:
            return [
                self.eta_min + (base_lr - self.eta_min) * (self.last_epoch / max(1, self.warmup_epochs))
                for base_lr in self.base_lrs
            ]
        # 2. 否则进入余弦退火阶段
        progress = float(self.last_epoch - self.warmup_epochs) / float(max(1, self.T_max - self.warmup_epochs))
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        return [
            self.eta_min + (base_lr - self.eta_min) * cosine_decay
            for base_lr in self.base_lrs
        ]

# ---------------------------------------------------------------------------
# 自定义调度器：带可选热身的 Cosine Annealing Warm Restarts (SGDR)
# ---------------------------------------------------------------------------

class CosineWarmRestartLR(LRScheduler):
    """
    带可选线性热身的余弦退火周期重启调度器 (SGDR)。

    热身阶段结束后按 CosineAnnealingWarmRestarts 公式内联计算学习率，
    周期计数以热身结束后的 epoch 为基准。

    参数
    ----
    optimizer      : Optimizer
    T_0            : int    首次重启周期长度（热身后的 epoch 数）
    T_mult         : int    每次重启后周期倍增系数，默认 1
    eta_min        : float  最低学习率，默认 0.0
    warmup_epochs  : int    线性热身轮数，默认 0（不热身）
    last_epoch     : int    同 PyTorch 惯例，默认 -1
    """

    def __init__(
        self,
        optimizer,
        T_0: int,
        T_mult: int = 1,
        eta_min: float = 0.0,
        warmup_epochs: int = 0,
        last_epoch: int = -1,
    ):
        self.T_0 = int(T_0)
        self.T_mult = int(T_mult)
        self.eta_min = float(eta_min)
        self.warmup_epochs = int(warmup_epochs)
        super().__init__(optimizer, last_epoch)

    @staticmethod
    def _sgdr_lr(t: int, T_0: int, T_mult: int, base_lr: float, eta_min: float) -> float:
        """给定热身后的 epoch t，内联计算 SGDR 学习率。"""
        T_i = T_0
        while t >= T_i:
            t -= T_i
            T_i = max(1, T_i * T_mult)
        return eta_min + (base_lr - eta_min) * 0.5 * (1.0 + math.cos(math.pi * t / T_i))

    def get_lr(self):
        epoch = self.last_epoch
        if epoch < self.warmup_epochs:
            # 线性热身：从 eta_min 线性爬升到 base_lr
            ratio = epoch / max(1, self.warmup_epochs)
            return [
                self.eta_min + (base_lr - self.eta_min) * ratio
                for base_lr in self.base_lrs
            ]
        # 热身结束后，内联 SGDR 公式
        t = epoch - self.warmup_epochs
        return [
            self._sgdr_lr(t, self.T_0, self.T_mult, base_lr, self.eta_min)
            for base_lr in self.base_lrs
        ]

# ---------------------------------------------------------------------------
# 自定义调度器：线性热身 + ReduceLROnPlateau 自适应衰减
# ---------------------------------------------------------------------------

class WarmupPlateauLR(LRScheduler):
    """
    线性热身阶段结束后，交由 ReduceLROnPlateau 进行自适应衰减。

    热身期间：lr 从 eta_min 线性爬升到 base_lr。
    热身结束后：内部 ReduceLROnPlateau 实例接管，监控 val_loss 自适应降 lr。

    参数
    ----
    optimizer      : Optimizer
    warmup_epochs  : int    线性热身轮数
    mode           : str    "min" 或 "max"，默认 "min"
    factor         : float  每次衰减的乘数，默认 0.5
    patience       : int    无改善多少 epoch 后触发衰减，默认 10
    min_lr         : float  lr 下界，默认 1e-6
    eta_min        : float  热身起始 lr，默认 0.0
    last_epoch     : int    同 PyTorch 惯例，默认 -1
    """

    def __init__(
        self,
        optimizer,
        warmup_epochs: int,
        mode: str = "min",
        factor: float = 0.5,
        patience: int = 10,
        min_lr: float = 1e-6,
        eta_min: float = 0.0,
        last_epoch: int = -1,
    ):
        self.warmup_epochs = int(warmup_epochs)
        self.eta_min = float(eta_min)
        # 热身结束后接管的 plateau 调度器（延迟到 super().__init__ 之后创建，
        # 此时 optimizer.param_groups 已就绪）
        self._plateau_kwargs = dict(
            mode=mode, factor=factor, patience=patience, min_lr=min_lr
        )
        self._plateau: ReduceLROnPlateau | None = None
        super().__init__(optimizer, last_epoch)

    def _get_plateau(self) -> ReduceLROnPlateau:
        if self._plateau is None:
            self._plateau = ReduceLROnPlateau(self.optimizer, **self._plateau_kwargs)
        return self._plateau

    def get_lr(self):
        # 热身阶段：线性爬升
        if self.last_epoch < self.warmup_epochs:
            ratio = self.last_epoch / max(1, self.warmup_epochs)
            return [
                self.eta_min + (base_lr - self.eta_min) * ratio
                for base_lr in self.base_lrs
            ]
        # 热身结束后：直接返回 optimizer 当前 lr（由 plateau 负责修改）
        return [pg["lr"] for pg in self.optimizer.param_groups]

    def step(self, metrics=None):  # type: ignore[override]
        if self.last_epoch < self.warmup_epochs:
            # 热身阶段走父类逻辑（调用 get_lr 并写入 optimizer）
            super().step()
        else:
            # 热身结束：epoch 计数仍需推进，但 lr 由 plateau 管理
            self.last_epoch += 1
            if metrics is not None:
                self._get_plateau().step(metrics)

    def state_dict(self):
        state = super().state_dict()
        state["_plateau_kwargs"] = self._plateau_kwargs
        if self._plateau is not None:
            state["_plateau_state"] = self._plateau.state_dict()
        return state

    def load_state_dict(self, state_dict):
        plateau_state = state_dict.pop("_plateau_state", None)
        self._plateau_kwargs = state_dict.pop("_plateau_kwargs", self._plateau_kwargs)
        super().load_state_dict(state_dict)
        if plateau_state is not None:
            self._get_plateau().load_state_dict(plateau_state)
