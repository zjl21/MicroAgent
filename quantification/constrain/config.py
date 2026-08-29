import json
import os
from glob import glob

from .model import MODEL_CONSTRAINTS, validate_model_constraints


def build_constraint_context(task_dir, test=False):
    pattern = os.path.join(task_dir, "*_features.json") if test else os.path.join(task_dir, "data", "*_features.json")
    features_path = glob(pattern)[0]
    with open(features_path, "r", encoding="utf-8") as f:
        features = json.load(f)
    return {
        "features_path": features_path,
        "n_shells": features["gradient"]["num_shells"] + 1,
    }


def _manual_shell_weights(config, context):
    dc_cfg = config.get("loss", {}).get("data_consistency", {})
    if not isinstance(dc_cfg, dict) or dc_cfg.get("shell_weighting") != "manual":
        return True, "Valid"

    n_shells = context.get("n_shells")
    shell_weights = dc_cfg.get("shell_weights", [])
    if len(shell_weights) != n_shells:
        return False, (
            f"shell_weighting='manual' requires exactly {n_shells} entries in shell_weights "
            f"(one per b-shell), but got {len(shell_weights)}."
        )
    return True, "Valid"


CONFIG_CONSTRAINTS = [
    {
        "name": "manual shell weights",
        "when": "loss.data_consistency.shell_weighting == 'manual'",
        "require": "len(loss.data_consistency.shell_weights) == n_shells",
        "check": _manual_shell_weights,
    },
    {
        "name": "model constraints",
        "when": "always",
        "require": MODEL_CONSTRAINTS,
        "check": lambda config, context: validate_model_constraints(config),
    },
]


def validate_config_constraints(config, context=None):
    context = context or {}
    for rule in CONFIG_CONSTRAINTS:
        is_valid, msg = rule["check"](config, context)
        if not is_valid:
            return False, msg
    return True, "Valid"
