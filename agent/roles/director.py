import os
import sys
import json
import subprocess
import numpy as np

from agent.tools.grad import run_diagnose_grad, has_nan_or_inf, is_loss_flat
from agent.tools.reward import RewardEvaluator
from agent.tools.llm import call_llm
from agent.tools.utility import extract_tag, compress_history
from agent.prompts.director import DirectorPrompt
from agent.tools.gpu_holder import GPUHolder, GPU_SLOT_UNAVAILABLE
from toolbox.utility.paths import project_path


class Director:

    def __init__(self, task_dir: str = None, physics: str = "DTI", gpu_holder: GPUHolder = None, quiet: bool = False):
        self.task_dir = task_dir
        self.api_config_path = os.path.join(os.path.dirname(task_dir), "api.json") if task_dir else None
        self.physics = physics
        self.past_records = []
        self.gpu_holder = gpu_holder
        self.quiet = quiet

    # ── 公开接口 ──────────────────────────────────────────────────────────

    def evaluate(self, trial_dir: str) -> str:
        """
        评估刚跑完的训练结果，决定下一步行动。
        返回 architect_instructions。
        """
        self.trial_dir   = trial_dir
        self.trial_num   = os.path.basename(trial_dir).split("_")[-1]
        self.config_path = os.path.join(trial_dir, "config.yaml")
        self._load_training_log()

        extra_info = ""

        if self.check_grad:
            if not self.quiet:
                print("🔍 [系统] 检测到 NaN/Inf 或 loss 完全平坦，自动启动梯度诊断...")
            extra_info = self._get_grad_info()
        else:
            if not self.quiet:
                print("✅ [Director] 梯度正常，准备评估 Reward 指标...")
            extra_info = self._get_reward_info()

        return self._call_llm(extra_info)

    # ── 私有方法 ──────────────────────────────────────────────────────────

    def _load_training_log(self):
        log_path = os.path.join(self.trial_dir, "training_log.json")
        try:
            with open(log_path, "r", encoding="utf-8") as f:
                log_data = json.load(f)
        except Exception as e:
            raise RuntimeError(f"[Director] 无法读取训练日志: {e}")

        full_history = log_data.get("training_history", [])
        if not full_history:
            raise RuntimeError("[Director] 训练日志为空，无法评估。")
        self.check_grad = has_nan_or_inf(full_history) or is_loss_flat(full_history)
        self.history = compress_history(full_history)

    def _ensure_gpu_hold(self, stage: str):
        if self.gpu_holder is None:
            return
        if getattr(self.gpu_holder, "tensor", None) is not None:
            return
        if not self.gpu_holder.occupy_on_device():
            raise RuntimeError(
                f"{GPU_SLOT_UNAVAILABLE}: failed to re-hold GPU {self.gpu_holder.device_id} "
                f"before Director {stage}"
            )

    def _get_grad_info(self) -> str:
        try:
            grad_out = os.path.join(self.trial_dir, "grad_diagnosis.json")
            run_diagnose_grad(self.config_path, gpu_holder=self.gpu_holder)
            self._ensure_gpu_hold("LLM analysis")
            with open(grad_out, "r", encoding="utf-8") as f:
                return f"\n\n【梯度诊断结果】\n{f.read()}"
        except Exception as e:
            raise RuntimeError(f"\n\n【梯度诊断失败】: {e}")

    def _get_reward_info(self) -> str:
        try:
            infer_script = project_path("infer.py")
            infer_stdout = os.path.join(self.trial_dir, "infer_stdout.log")
            infer_stderr = os.path.join(self.trial_dir, "infer_stderr.log")
            if not self.quiet:
                print(f"⏳ [系统] 正在执行全脑推理 (infer.py)，stdout: {infer_stdout}, stderr: {infer_stderr}")
            env = os.environ.copy()
            if self.gpu_holder is not None:
                env["CUDA_VISIBLE_DEVICES"] = str(self.gpu_holder.device_id)
            with open(infer_stdout, "w", encoding="utf-8") as f_out, open(infer_stderr, "w", encoding="utf-8") as f_err:
                try:
                    subprocess.run(
                        [sys.executable, infer_script, "--config", self.config_path],
                        stdout=f_out,
                        stderr=f_err,
                        check=True,
                        env=env,
                    )
                except subprocess.CalledProcessError as e:
                    err_content = ""
                    if os.path.exists(infer_stderr):
                        with open(infer_stderr, "r", encoding="utf-8") as f:
                            err_content = f.read()
                    if GPU_SLOT_UNAVAILABLE in err_content:
                        raise RuntimeError(err_content) from e
                    raise
            self._ensure_gpu_hold("reward evaluation")
            if not self.quiet:
                print("✅ [系统] 全脑推理完毕，开始计算 Reward 指标...")

            gt_dir = os.path.join(os.path.dirname(self.trial_dir), "ground_truth")
            RewardEvaluator(self.trial_dir, gt_dir=gt_dir).get_reward_summary()
 
            # 记录已有 reward 的 trial 编号
            if self.task_dir:
                reward_idx_path = os.path.join(self.task_dir, "reward_idx.txt")
                reward_idx = np.loadtxt(reward_idx_path) if os.path.exists(reward_idx_path) else np.array([])
                reward_idx_new = np.unique(np.append(reward_idx, int(self.trial_num)))  # 添加当前 trial_num 并去重
                np.savetxt(reward_idx_path, reward_idx_new, fmt="%d")

            with open(os.path.join(self.trial_dir, "reward_summary.json"), "r", encoding="utf-8") as f:
                return f"\n\n【Reward 评估结果】\n{f.read()}"
        except Exception as e:
            raise RuntimeError(f"\n\n【Reward 评估失败】: {e}")

    def _call_llm(self, extra_info: str) -> str:
        with open(self.config_path, "r", encoding="utf-8") as f:
            config_content = f.read()

        system_prompt = DirectorPrompt.build_system(self.check_grad)
        user_prompt   = DirectorPrompt.build_user(
            config_content, self.history, extra_info
        )

        if not self.quiet:
            print("\n🧠 [LLM 策略端] 正在对本轮训练结果进行综合研判，请稍候...")
        output_text = None
        try:
            output_text = call_llm(
                role="director",
                system_prompt=system_prompt,
                messages=[{"role": "user", "content": user_prompt}],
                api_config_path=self.api_config_path,
                temperature=0.3,
                max_tokens=6000,
                gpu_holder=self.gpu_holder,
            )
        except Exception as e:
            if GPU_SLOT_UNAVAILABLE in str(e):
                raise
            raise RuntimeError("[Director] API 多次调用失败，终止本轮评估。")

        if output_text is None:
            raise RuntimeError("[Director] API 多次调用失败，终止本轮评估。")

        summary = extract_tag(output_text, "summary")
        failure_diagnosis = extract_tag(output_text, "failure_diagnosis")

        if not self.quiet:
            print(f"💡 [LLM 策略端] 诊断完成")

        return summary, failure_diagnosis
