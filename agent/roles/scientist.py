import os
import json
import numpy as np

from agent.tools.llm import call_llm
from agent.tools.skill import SkillManager
from agent.prompts.scientist import ScientistPrompt
from agent.tools.reward import compute_score
from agent.prompts.architect import ArchitectPrompt
from agent.runtime import configure_library_path
configure_library_path()

class Scientist:
    """科学家角色：task 完成后复盘，更新 skill.json"""

    def __init__(self, skill_manager: SkillManager, quiet: bool = False):
        self.skill_manager = skill_manager
        self.api_config_path = os.path.join(os.path.dirname(skill_manager.skill_path), "api.json")
        self.quiet = quiet

    def reflect(self, task_dir: str, summary: str, gpu_holder=None):
        """对完成的 task 进行复盘"""
        if not self.quiet:
            print(f"\n🔬 [Scientist] 开始复盘 {os.path.basename(task_dir)}...")

        # 读取 architect_state.json 获取完整信息
        architect_state_path = os.path.join(task_dir, "architect_state.json")
        if os.path.exists(architect_state_path):
            with open(architect_state_path, "r", encoding="utf-8") as f:
                architect_records = json.load(f)
        else:
            architect_records = []

        system_prompt = ScientistPrompt.build_system_task()
        user_prompt = ScientistPrompt.build_user_task(architect_records, summary)

        output = call_llm(
            role="scientist",
            system_prompt=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
            api_config_path=self.api_config_path,
            temperature=0.3,
            max_tokens=8192,
            gpu_holder=gpu_holder,
        )

        draft = self.skill_manager.parse_task_skill(output)
        draft_path = os.path.join(task_dir, "skill.json")
        with open(draft_path, "w", encoding="utf-8") as f:
            json.dump(draft, f, ensure_ascii=False, indent=2)
        if not self.quiet:
            print(f"✅ [Scientist] 复盘完成")

    def select_success(self, task_dir: str):
        """选择与最好 trial 分数相差 5% 以内的方案。"""
        fpIdx = os.path.join(task_dir,'reward_idx.txt')
        idx_all = np.loadtxt(fpIdx, dtype=int, ndmin=1)
        metric_all = np.zeros_like(idx_all, dtype=float)

        for ii, idx in enumerate(idx_all):
            fpReward = os.path.join(task_dir, f"trial_{idx}", "reward_summary.json")
            reward = json.load(open(fpReward,'r',encoding='utf-8'))
            metric, sort_method = compute_score(reward)
            metric_all[ii] = metric

        if sort_method == 'ASCENDING':
            best_metric = np.max(metric_all)
            metric_thr = best_metric * 0.95
            best_idx = idx_all[metric_all >= metric_thr]
        else:
            best_metric = np.min(metric_all)
            metric_thr = best_metric * 1.05
            best_idx = idx_all[metric_all <= metric_thr]

        fpArc_State = os.path.join(task_dir, "architect_state.json")
        arc_state = json.load(open(fpArc_State,'r',encoding='utf-8'))

        exp_dir = os.path.dirname(task_dir)
        dpSuccess = os.path.join(exp_dir, "success")
        os.makedirs(dpSuccess, exist_ok=True)
        fpExp = os.path.join(dpSuccess, f"{os.path.basename(task_dir)}.json")
        exp_all = {}
        exp_all['task'] = ArchitectPrompt.build_task_description(exp_dir, int(os.path.basename(task_dir).split("_")[-1]))
        exp_all['experiences'] = []

        for ii, idx in enumerate(best_idx):
            exp_tmp = {}
            config = arc_state[idx-1].get("config", {})
            score = metric_all[np.where(idx_all == idx)[0][0]]
            exp_tmp['trial'] = int(idx)
            exp_tmp['config'] = config
            exp_tmp['score'] = float(score)
            exp_all['experiences'].append(exp_tmp)

        with open(fpExp, "w", encoding="utf-8") as f:
            json.dump(exp_all, f, ensure_ascii=False, indent=2)

    def reflect_success(self):
        """读取所有 success/*.json，提炼跨任务经验和任务特定的参数规律，更新 skill.json"""
        exp_dir = os.path.dirname(self.skill_manager.skill_path)
        dpSuccess = os.path.join(exp_dir, "success")
        if not os.path.isdir(dpSuccess):
            if not self.quiet:
                print("[Scientist] No success directory found, skipping reflect_success.")
            return

        success_files = sorted(
            [f for f in os.listdir(dpSuccess) if f.endswith(".json")],
            key=lambda name: int(os.path.splitext(name)[0].split("_")[-1]),
        )
        if not success_files:
            if not self.quiet:
                print("[Scientist] No success records found, skipping reflect_success.")
            return

        # 跨任务 skill 只使用任务特征和 task-level contrast；成功配置由 Architect 单独使用。
        all_records = []
        for fname in success_files:
            fp = os.path.join(dpSuccess, fname)
            with open(fp, "r", encoding="utf-8") as f:
                success_record = json.load(f)
            task_name = os.path.splitext(fname)[0]
            task_dir = os.path.join(exp_dir, task_name)
            record = {
                "task_idx": int(task_name.split("_")[-1]),
                "task": success_record.get("task"),
            }

            draft_path = os.path.join(task_dir, "skill.json")
            # Portable releases can contain successful anchors without the
            # original task directories. They are valid Architect context but
            # cannot participate in a fresh task-skill integration pass.
            if not os.path.isfile(draft_path):
                continue
            with open(draft_path, "r", encoding="utf-8") as f:
                record["task_skill"] = json.load(f)

            all_records.append(record)

        if not all_records:
            if not self.quiet:
                print("[Scientist] No task-level skill drafts found; skipping integration.")
            return

        system_prompt = ScientistPrompt.build_system_integrate()
        user_prompt = ScientistPrompt.build_user_integrate(all_records)

        if not self.quiet:
            print(f"\n🔬 [Scientist] 从 {len(all_records)} 个成功任务中提炼全局经验...")
        output = call_llm(
            role="scientist",
            system_prompt=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
            api_config_path=self.api_config_path,
            temperature=0.3,
            max_tokens=163840,
        )

        fpOut = os.path.join(exp_dir, "skill.json")
        with open(fpOut, "w", encoding="utf-8") as f:
            f.write(output)
            
        if not self.quiet:
            print("✅ [Scientist] 全局经验提炼完成")
