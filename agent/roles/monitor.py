import os
import sys
import json
import time
import subprocess

from agent.tools.llm import call_llm
from agent.tools.gpu_holder import GPU_SLOT_UNAVAILABLE, query_gpu_memory

_PROMPT_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "prompts", "monitor.md")

# run() 的返回值含义
DONE  = "DONE"   # 正常结束或被 Monitor 叫停 → 交给 Director 复盘
OOM   = "OOM"    # CUDA out of memory → 通知 Architect 减内存参数
ERROR = "ERROR"  # 脚本级别报错 → 终止实验流
REQUEUE = "REQUEUE"  # GPU slot handoff failed → keep task state and retry later


class Monitor:
    """
    负责启动训练子进程并在运行期间持续监控。

    用法::

        monitor = Monitor()
        status, msg = monitor.run(trial_dir)
    """

    def __init__(self, check_interval_sec: int = 20, gpu_holder=None, quiet: bool = False, enable_holder: bool = False):
        self.check_interval_sec = check_interval_sec
        self.gpu_holder = gpu_holder
        self.quiet = quiet
        self.enable_holder = enable_holder
        with open(_PROMPT_PATH, "r", encoding="utf-8") as f:
            self._system_prompt = f.read()

    # ── 公开接口 ──────────────────────────────────────────────────────────

    def run(self, trial_dir: str) -> str:
        """
        启动训练并监控，直到进程结束。

        Returns:
            DONE          训练正常完毕，或被 Monitor 提前叫停
            OOM           CUDA 显存不足
            ERROR         脚本异常退出
        """
        log_path    = os.path.join(trial_dir, "training_log.json")
        stdout_path = os.path.join(trial_dir, "stdout.log")
        stderr_path = os.path.join(trial_dir, "stderr.log")
        config_path = os.path.join(trial_dir, "config.yaml")

        if not self.quiet:
            print(f"🚀 启动训练进程 (后台挂起)，stdout: {stdout_path}, stderr: {stderr_path}")
        f_out = open(stdout_path, "w", encoding="utf-8")
        f_err = open(stderr_path, "w", encoding="utf-8")
        try:
            env = os.environ.copy()
            if self.gpu_holder is not None:
                env["CUDA_VISIBLE_DEVICES"] = str(self.gpu_holder.device_id)
            process = subprocess.Popen(
                [sys.executable, "train.py", "--config", config_path],
                stdout=f_out,
                stderr=f_err,
                env=env,
            )
            self._watch(log_path, process)
            if process.poll() is None:
                process.wait()
        finally:
            f_out.close()
            f_err.close()

        return self._check_result(process, stderr_path, log_path)

    # ── 私有方法 ──────────────────────────────────────────────────────────

    def _watch(self, log_path: str, process: subprocess.Popen):
        """LLM 巡防监视循环，每间隔一定时间向 LLM 问询一次。"""
        last_seen_epoch = -1
        gpu_occupied = False

        while process.poll() is None:
            time.sleep(10)
            if process.poll() is not None:
                break

            if not os.path.exists(log_path):
                continue
            try:
                with open(log_path, "r", encoding="utf-8") as f:
                    log_data = json.load(f)
            except Exception:
                continue  # 原子替换期间极小概率的窗口，下一轮再试

            history = log_data.get("training_history", [])
            if not history:
                continue

            # 至少完成一个 epoch 后再占位，避免在 CUDA/cuBLAS 初始化和首轮 forward 前抢占显存。
            if self.enable_holder and not gpu_occupied and self.gpu_holder:
                self.gpu_holder.occupy_on_device()
                gpu_occupied = True

            current_epoch = history[-1].get("epoch", -1)
            if current_epoch == last_seen_epoch:
                continue  # 没有新 epoch，不必重复问询

            # print(f"\n🤖 [LLM 监控员] 正在根据完整记录评估 Epoch {current_epoch} 的健康状态...")

            # try:
            #     decision_text = call_llm(
            #         role="monitor",
            #         system_prompt=self._system_prompt,
            #         messages=[{"role": "user", "content": f"完整训练记录:\n{json.dumps(history, indent=2)}\n\n请指示: STOP 或 CONTINUE。"}],
            #         temperature=0.0,
            #         max_tokens=1024,
            #     )
            #     decision = decision_text.strip().upper()

            #     if "STOP" in decision:
            #         print(f"🚨 [LLM 监控员] '{decision}'！立马终止训练进程。")
            #         process.terminate()
            #         process.wait()
            #         self._write_stop_marker(log_path, log_data)
            #         break
            #     else:
            #         print(f"✅ [LLM 监控员]'{decision}'，训练继续。")

            # except Exception as e:
            #     print(f"⚠️ [LLM 监控员] API 调用故障，默认放行... ({e})")

            last_seen_epoch = current_epoch
            time.sleep(self.check_interval_sec)

    def _write_stop_marker(self, log_path: str, fallback_log_data: dict):
        """在日志里写入提前终止标记，供 Director 识别。"""
        try:
            # 重新读取文件，获取进程停止后的最终状态
            # （避免覆盖 LLM 推理期间 train.py 新写入的 epoch）
            try:
                with open(log_path, "r", encoding="utf-8") as f:
                    log_data = json.load(f)
            except Exception:
                log_data = fallback_log_data

            log_data["summary"] = {
                "stopped_early": True,
                "reason": "Killed by Monitor due to non-convergence.",
            }
            with open(log_path, "w", encoding="utf-8") as f:
                json.dump(log_data, f, indent=2)
        except Exception as e:
            if not self.quiet:
                print(f"⚠️ 保存终止日志失败: {e}")

    def _mark_failure(self, log_path: str, status: str, reason: str, stderr_path: str, returncode: int):
        try:
            if os.path.exists(log_path):
                with open(log_path, "r", encoding="utf-8") as f:
                    log_data = json.load(f)
            else:
                log_data = {"training_history": [], "summary": {}}
            log_data["summary"] = {
                **log_data.get("summary", {}),
                "status": status,
                "reason": reason,
                "returncode": returncode,
                "stderr_path": stderr_path,
            }
            with open(log_path, "w", encoding="utf-8") as f:
                json.dump(log_data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            if not self.quiet:
                print(f"⚠️ 保存失败日志失败: {e}")

    def _gpu_has_training_budget(self):
        if self.gpu_holder is None or self.gpu_holder.hold_mib is None:
            return False
        gpu_id = self.gpu_holder.device_id
        free_mib, _ = query_gpu_memory([gpu_id]).get(gpu_id, (0, 0))
        return free_mib >= int(self.gpu_holder.hold_mib)

    def _check_result(self, process: subprocess.Popen, stderr_path: str, log_path: str) -> tuple:
        """根据子进程退出码判断本轮训练结果。"""
        returncode = process.returncode

        if returncode == 0:
            return DONE

        if returncode < 0:
            # 负值 = 被信号杀死（Monitor terminate() 发的是 SIGTERM = -15）
            if not self.quiet:
                print(f"⚠️ 训练进程被 Monitor 强制终止 (信号: {-returncode})，准备进入 Director 复盘并调整参数！")
            self._mark_failure(log_path, "STOPPED", f"terminated by signal {-returncode}", stderr_path, returncode)
            return DONE

        # 正的非零退出码 = 脚本报错
        with open(stderr_path, "r", encoding="utf-8") as f:
            err_content = f.read()

        oom_signatures = [
            "CUDA out of memory",
            "CUDA error: out of memory",
            "CUBLAS_STATUS_ALLOC_FAILED",
            "cublasCreate(handle)",
        ]
        if GPU_SLOT_UNAVAILABLE in err_content:
            if self._gpu_has_training_budget():
                self._mark_failure(
                    log_path,
                    "OOM",
                    "GPU slot was available after handoff failure; treating as model memory overflow",
                    stderr_path,
                    returncode,
                )
                return OOM
            self._mark_failure(log_path, "REQUEUE", "GPU slot was taken during handoff", stderr_path, returncode)
            return REQUEUE

        if any(sig in err_content for sig in oom_signatures):
            if not self.quiet:
                print("🔥 检测到爆显存 (CUDA out of memory) 报错！将通知 Architect 减小 batch_size 或 patch_size。")
            self._mark_failure(log_path, "OOM", "CUDA memory allocation failed", stderr_path, returncode)
            return OOM

        if not self.quiet:
            print(f"❌ 训练进程因代码报错异常退出 (返回码: {returncode})，终止自动实验流！\n日志查看: {stderr_path}")
        if err_content.strip() and not self.quiet:
            print("------ 报错信息尾部 ------")
            print("\n".join(err_content.strip().split("\n")[-15:]))
        self._mark_failure(log_path, "ERROR", "training process exited with a non-zero code", stderr_path, returncode)
        return ERROR
