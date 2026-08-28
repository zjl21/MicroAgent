import json
import os

import numpy as np
from toolbox.utility.iolib import load_config

from agent.runtime import build_model, configure_library_path

configure_library_path()

from reward import collect_metrics, compute_score


class RewardEvaluator:
    def __init__(self, trial_dir: str, gt_dir: str = None):
        self.trial_dir = trial_dir
        self.gt_dir = gt_dir
        self.pred_dir = os.path.join(trial_dir, "output")
        self.config = load_config(os.path.join(trial_dir, "config.yaml"))
        self.physics = self.config["physics"]["name"]
        self.task_dir = os.path.dirname(trial_dir)
        self.data_dir = self.config["data"]["data_dir"]

        trial_num = os.path.basename(trial_dir).split("_")[-1]
        self.reward_summary = {"trial_num": trial_num}
        self.history_rewards = self._load_history_rewards(trial_num)
        self.history = self._load_training_history()

        self.model = build_model(self.config)

    def _load_history_rewards(self, trial_num):
        fp_idx = os.path.join(self.task_dir, "reward_idx.txt")
        idxs = np.loadtxt(fp_idx, dtype=int, ndmin=1) if os.path.exists(fp_idx) else []
        idxs = [idx for idx in np.unique(idxs) if idx != int(trial_num)]
        rewards = []
        for idx in idxs:
            fp_reward = os.path.join(self.task_dir, f"trial_{idx}", "reward_summary.json")
            if os.path.exists(fp_reward):
                with open(fp_reward, "r", encoding="utf-8") as f:
                    rewards.append(json.load(f))
        return rewards

    def _load_training_history(self):
        fp_history = os.path.join(self.trial_dir, "training_log.json")
        if not os.path.exists(fp_history):
            raise FileNotFoundError(f"Training log not found: {fp_history}")
        with open(fp_history, "r", encoding="utf-8") as f:
            return json.load(f).get("training_history", [])

    def _with_history(self, value, metric_path, higher_better=True, ideal_value=None):
        if value is None:
            return {"value": None, "better": "unavailable"}
        if not self.history_rewards:
            better_desc = f"closer to {ideal_value}" if ideal_value is not None else ("higher" if higher_better else "lower")
            return {"value": round(float(value), 4), "better": better_desc}

        hist_vals = []
        for reward in self.history_rewards:
            v = self._get_metric_value(reward, metric_path)
            if v is not None:
                hist_vals.append(v)
        if not hist_vals:
            better_desc = f"closer to {ideal_value}" if ideal_value is not None else ("higher" if higher_better else "lower")
            return {"value": round(float(value), 4), "better": better_desc}

        if ideal_value is not None:
            deviations = [abs(v - ideal_value) for v in hist_vals]
            dev = abs(float(value) - ideal_value)
            dev_best = min(deviations)
            mean_dev = np.mean(deviations)
            dev_last = deviations[-1]
            is_best = dev < dev_best
            better_desc = f"closer to {ideal_value}"
            vs_best = dev / dev_best if dev_best != 0 else 0
            vs_mean = dev / mean_dev if mean_dev != 0 else 0
            vs_last = dev / dev_last if dev_last != 0 else 0
            all_devs = [dev] + deviations
            min_dev, max_dev = min(all_devs), max(all_devs)
            norm_value = 1 - (dev - min_dev) / (max_dev - min_dev) if max_dev != min_dev else 1
        else:
            hist_best = max(hist_vals) if higher_better else min(hist_vals)
            hist_mean = np.mean(hist_vals)
            hist_last = hist_vals[-1]
            all_vals = [float(value)] + hist_vals
            norm_value = (float(value) - min(all_vals)) / (max(all_vals) - min(all_vals)) if max(all_vals) != min(all_vals) else 0
            norm_value = norm_value if higher_better else 1 - norm_value
            is_best = value > hist_best if higher_better else value < hist_best
            better_desc = "higher" if higher_better else "lower"
            vs_best = float(value) / hist_best if hist_best != 0 else 0
            vs_mean = float(value) / hist_mean if hist_mean != 0 else 0
            vs_last = (float(value) - hist_last) / abs(hist_last) if hist_last != 0 else 0

        return {
            "value": round(float(value), 4),
            "vs_best": round(vs_best, 4),
            "vs_mean": round(vs_mean, 4),
            "vs_last": round(vs_last, 4),
            "is_best": is_best,
            "better": better_desc,
            "norm_value": round(norm_value, 4),
        }

    @staticmethod
    def _get_metric_value(reward, metric_path):
        v = reward
        for key in metric_path:
            if not isinstance(v, dict) or key not in v:
                return None
            v = v[key]
        if isinstance(v, dict):
            v = v.get("value")
        return v if isinstance(v, (int, float)) else None

    @staticmethod
    def _set_nested(target, path, value):
        cur = target
        for key in path[:-1]:
            cur = cur.setdefault(key, {})
        cur[path[-1]] = value

    def get_reward_summary(self):
        for spec in collect_metrics(self):
            path = spec["path"]
            value = spec["value"]
            metric = self._with_history(
                value,
                path,
                spec.get("higher_better", True),
                spec.get("ideal_value"),
            )
            self._set_nested(self.reward_summary, path, metric)

        summary_path = os.path.join(self.trial_dir, "reward_summary.json")
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(self.reward_summary, f, indent=4)
