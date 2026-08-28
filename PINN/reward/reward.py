import os
from glob import glob

import nibabel as nib
import numpy as np
from scipy import stats


def compute_score(reward):
    mean_rel_mae = reward["param_quality"]["mean_rel_MAE"]
    score = mean_rel_mae["value"] if isinstance(mean_rel_mae, dict) else mean_rel_mae
    return score, "DESCENDING"


def collect_metrics(evaluator):
    metrics = []
    metrics.extend(_convergence_metrics(evaluator))
    metrics.extend(_complexity_metrics(evaluator))
    metrics.extend(_robustness_metrics(evaluator))
    metrics.extend(_map_quality_metrics(evaluator))
    return metrics


def _metric(path, value, higher_better=True, ideal_value=None):
    return {
        "path": path,
        "value": value,
        "higher_better": higher_better,
        "ideal_value": ideal_value,
    }


def _convergence_metrics(evaluator):
    history = evaluator.history
    if not history:
        return []

    total_epochs = history[-1]["epoch"]
    min_mse = min(
        e["val_comps"]["MSE_unweighted"]
        for e in history
        if isinstance(e.get("val_comps", {}).get("MSE_unweighted"), (int, float))
    )
    threshold_mse = min_mse * 1.05
    epoch_mse = next(
        e["epoch"]
        for e in history
        if isinstance(e.get("val_comps", {}).get("MSE_unweighted"), (int, float))
        and e["val_comps"]["MSE_unweighted"] <= threshold_mse
    )

    min_loss = min(e["val_loss"] for e in history)
    threshold_loss = min_loss * 1.05
    epoch_loss = next(e["epoch"] for e in history if e["val_loss"] <= threshold_loss)

    ratio_loss = epoch_loss / total_epochs if total_epochs else None
    ratio_mse = epoch_mse / total_epochs if total_epochs else None

    return [
        _metric(["convergence_efficiency", "epoch_number"], total_epochs, False),
        _metric(["convergence_efficiency", "min_val_loss"], min_loss, False),
        _metric(["convergence_efficiency", "min_MSE_loss"], min_mse, False),
        _metric(["convergence_efficiency", "convergence_epoch_val_loss"], epoch_loss, False),
        _metric(["convergence_efficiency", "convergence_epoch_MSE_loss"], epoch_mse, False),
        _metric(["convergence_efficiency", "convergence_ratio_val_loss"], ratio_loss, True),
        _metric(["convergence_efficiency", "convergence_ratio_MSE_loss"], ratio_mse, True),
    ]


def _complexity_metrics(evaluator):
    trainable_params = sum(p.numel() for p in evaluator.model.parameters() if p.requires_grad)
    return [_metric(["complexity", "trainable_params"], trainable_params, False)]


def _robustness_metrics(evaluator):
    losses = [e["val_loss"] for e in evaluator.history]
    if len(losses) < 2:
        return [_metric(["robustness", "stability"], None, True)]
    deltas = np.abs(np.diff(losses))
    mean_loss = np.mean(losses)
    stability = float(np.clip(1 - np.std(deltas) / (mean_loss + 1e-12), 0, 1))
    return [_metric(["robustness", "stability"], stability, True)]


def _map_quality_metrics(evaluator):
    physics_map = {
        "DTI": {
            "FA": _eval_scalar,
            "MD": _eval_scalar,
            "L1": _eval_scalar,
            "RD": _eval_scalar,
            "V1": _eval_vector,
        },
        "DKI": {
            "FA": _eval_scalar,
            "MD": _eval_scalar,
            "L1": _eval_scalar,
            "RD": _eval_scalar,
            "AK": _eval_scalar_median,
            "RK": _eval_scalar_median,
            "MK": _eval_scalar_median,
        },
        "NODDI":{
            "odi": _eval_scalar,
            "ficvf": _eval_scalar,
            "fiso": _eval_scalar,
            "fiberdir": _eval_vector
        }
    }
    if evaluator.physics not in physics_map:
        raise ValueError(f"Unsupported physics: {evaluator.physics}. Available: {list(physics_map.keys())}")

    mask_path = glob(os.path.join(evaluator.data_dir, "*_evaluate_mask*.nii.gz"))[0]
    mask = nib.load(mask_path).get_fdata().astype(bool)

    metrics = []
    r2_list, rel_mae_list = [], []
    for param_name, eval_fn in physics_map[evaluator.physics].items():
        gt_path = glob(os.path.join(evaluator.gt_dir, f"*{param_name}*.nii.gz"))[0]
        pr_path = glob(os.path.join(evaluator.pred_dir, f"*{param_name}*.nii.gz"))[0]
        gt_v = nib.load(gt_path).get_fdata()[mask]
        pr_v = nib.load(pr_path).get_fdata()[mask]
        raw = eval_fn(gt_v, pr_v)

        for metric_name, value in raw.items():
            higher = metric_name in ["R2"]
            ideal = 0 if metric_name == "Bias" else (0.5 if metric_name == "pos_ratio" else None)
            metrics.append(_metric(["param_quality", "per_param", param_name, metric_name], value, higher, ideal))

        if "R2" in raw:
            r2_list.append(raw["R2"])
        if "rel_MAE" in raw:
            rel_mae_list.append(raw["rel_MAE"])

    mean_r2 = sum(r2_list) / len(r2_list) if r2_list else None
    mean_rel_mae = sum(rel_mae_list) / len(rel_mae_list) if rel_mae_list else None
    metrics.append(_metric(["param_quality", "mean_R2"], mean_r2, True))
    metrics.append(_metric(["param_quality", "mean_rel_MAE"], mean_rel_mae, False))
    return metrics


def _eval_scalar(gt_v, pr_v):
    diff = pr_v - gt_v
    r_val, _ = stats.pearsonr(gt_v, pr_v)
    r2 = float(r_val ** 2) if not np.isnan(r_val) else 0.0
    mae = float(np.mean(np.abs(diff)))
    mean_gt = float(np.mean(gt_v)) or 1.0
    return {
        "R2": r2,
        "rel_MAE": mae / abs(mean_gt),
        "MAE": mae,
        "Bias": float(np.mean(diff)),
        "pos_ratio": float(np.mean(diff > 0)),
    }


def _eval_scalar_median(gt_v, pr_v):
    valid = np.isfinite(gt_v) & np.isfinite(pr_v)
    gt_v = gt_v[valid]
    pr_v = pr_v[valid]
    diff = pr_v - gt_v

    inlier = np.ones(gt_v.shape, dtype=bool)
    if gt_v.size >= 100:
        lower, upper = np.percentile(gt_v, [0.5, 99.5])
        if np.isfinite(lower) and np.isfinite(upper) and lower < upper:
            candidate = (gt_v >= lower) & (gt_v <= upper)
            if np.count_nonzero(candidate) >= 2:
                inlier = candidate

    gt_inlier = gt_v[inlier]
    pr_inlier = pr_v[inlier]
    diff_inlier = pr_v[inlier] - gt_v[inlier]
    if gt_inlier.size < 2 or np.all(gt_inlier == gt_inlier[0]) or np.all(pr_inlier == pr_inlier[0]):
        r2 = 0.0
    else:
        r_val, _ = stats.pearsonr(gt_inlier, pr_inlier)
        r2 = float(r_val ** 2) if not np.isnan(r_val) else 0.0

    mae = float(np.median(np.abs(diff)))
    mean_gt = float(np.median(gt_v)) or 1.0
    return {
        "R2": r2,
        "rel_MAE": mae / abs(mean_gt),
        "MAE": mae,
        "Bias": float(np.median(diff_inlier)),
        "pos_ratio": float(np.mean(diff_inlier > 0)),
    }


def _eval_vector(gt_v, pr_v):
    gt_norm = np.sqrt(gt_v[:, 0:1] * gt_v[:, 0:1] + gt_v[:, 1:2] * gt_v[:, 1:2] + gt_v[:, 2:3] * gt_v[:, 2:3])
    gt_norm[gt_norm == 0] = 1
    gt_v = gt_v / gt_norm
    pr_norm = np.sqrt(pr_v[:, 0:1] * pr_v[:, 0:1] + pr_v[:, 1:2] * pr_v[:, 1:2] + pr_v[:, 2:3] * pr_v[:, 2:3])
    pr_norm[pr_norm == 0] = 1
    pr_v = pr_v / pr_norm

    dotprod = abs(np.sum(gt_v * pr_v, axis=-1))
    dotprod[dotprod > 1] = 1
    ang_err = np.arccos(dotprod) / np.pi * 180
    return {
        "mean_angular_error_deg": float(np.mean(ang_err)),
        "median_angular_error_deg": float(np.median(ang_err)),
    }
