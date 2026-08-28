import numpy as np
from skimage.metrics import structural_similarity as ssim
import os
from glob import glob
import nibabel as nib

def _safe_round(value, ndigits=6):
    value = float(value)
    if not np.isfinite(value):
        return None
    return round(value, ndigits)


def array_stats(values, ndigits=6):
    values = np.asarray(values)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {"mean": None, "std": None, "p01": None, "p05": None, "p50": None, "p95": None, "p99": None}
    return {
        "mean": _safe_round(np.mean(values), ndigits),
        "std": _safe_round(np.std(values), ndigits),
        "p01": _safe_round(np.percentile(values, 1), ndigits),
        "p05": _safe_round(np.percentile(values, 5), ndigits),
        "p50": _safe_round(np.percentile(values, 50), ndigits),
        "p95": _safe_round(np.percentile(values, 95), ndigits),
        "p99": _safe_round(np.percentile(values, 99), ndigits),
    }


def _direction_uniformity(shell):
    norm = np.linalg.norm(shell, axis=1)
    shell = shell[norm > 1e-6]
    if shell.shape[0] < 2:
        return {"min_pairwise_angle_deg": None, "mean_nearest_angle_deg": None}

    shell = shell / np.linalg.norm(shell, axis=1, keepdims=True)
    cos = np.clip(np.abs(shell @ shell.T), 0.0, 1.0)
    tri = np.triu_indices(shell.shape[0], k=1)
    pair_angles = np.degrees(np.arccos(cos[tri]))

    cos_no_self = cos.copy()
    np.fill_diagonal(cos_no_self, -np.inf)
    nearest_angles = np.degrees(np.arccos(np.max(cos_no_self, axis=1)))
    return {
        "min_pairwise_angle_deg": _safe_round(np.min(pair_angles)),
        "mean_nearest_angle_deg": _safe_round(np.mean(nearest_angles)),
    }


def gradient_features(bval, bvec):
    bval = np.asarray(bval).reshape(-1)
    bvec = np.asarray(bvec)
    if bvec.ndim == 1:
        bvec = bvec.reshape(1, -1)
    if bvec.shape[-1] != 3 and bvec.shape[0] == 3:
        bvec = bvec.T

    out = {}
    bvals = np.sort(np.unique(bval))
    nonzero = bvals[bvals != 0]
    out["bvals"] = [int(v) for v in bvals]
    out["num_directions"] = int(bval.shape[0])
    out["num_shells"] = int(len(nonzero))
    out["b0_fraction"] = _safe_round(np.mean(bval == 0))
    out["max_bval"] = int(np.max(nonzero)) if nonzero.size else 0
    out["bval_span"] = int(np.max(nonzero) - np.min(nonzero)) if nonzero.size > 1 else 0

    for ub in bvals:
        idx = np.where(bval == ub)[0]
        out[f"num_b{int(ub)}"] = int(len(idx))
        out[f"fraction_b{int(ub)}"] = _safe_round(len(idx) / max(len(bval), 1))
        if ub != 0:
            shell = bvec[idx, :]
            gx, gy, gz = shell[:, 0], shell[:, 1], shell[:, 2]
            gradient = np.stack(
                [ub * gx * gx, ub * gy * gy, ub * gz * gz, 2 * ub * gx * gy, 2 * ub * gx * gz, 2 * ub * gy * gz],
                axis=1,
            )
            out[f"cond_b{int(ub)}"] = _safe_round(np.linalg.cond(gradient))
            out[f"rank_b{int(ub)}"] = int(np.linalg.matrix_rank(gradient))
            out[f"bvec_norm_b{int(ub)}"] = array_stats(np.linalg.norm(shell, axis=1))
            out.update({f"{k}_b{int(ub)}": v for k, v in _direction_uniformity(shell).items()})
    return out


def signal_shell_stats(signal, mask, bval, b0_image=None):
    signal = np.asarray(signal)
    mask = np.asarray(mask).astype(bool)
    bval = np.asarray(bval).reshape(-1)
    out = {}
    for ub in np.unique(bval).astype(int):
        shell = signal[..., bval == ub]
        shell_signal = shell[mask]
        shell_mean = np.mean(shell, axis=-1)
        entry = {
            "raw_signal": array_stats(shell_signal),
            "mean_image": array_stats(shell_mean[mask]),
            "coefficient_of_variation": _safe_round(np.std(shell_signal) / (np.mean(shell_signal) + 1e-12)),
            "negative_signal_fraction": _safe_round(np.mean(shell_signal < 0)),
        }
        if b0_image is not None:
            normalized = shell_mean[mask] / (b0_image[mask] + 1e-12)
            entry["normalized_to_b0"] = array_stats(normalized)
            entry["attenuation_mean"] = _safe_round(1.0 - np.mean(normalized))
            entry["low_signal_fraction"] = _safe_round(np.mean(normalized < 0.05))
        else:
            entry["low_signal_fraction"] = _safe_round(np.mean(shell_signal < 0.05 * (np.mean(shell_signal) + 1e-12)))
        out[str(ub)] = entry
    return out



def signal_pair_shell_metrics(reference, prediction, mask, bval):
    reference = np.asarray(reference)
    prediction = np.asarray(prediction)
    mask = np.asarray(mask).astype(bool)
    bval = np.asarray(bval).reshape(-1)

    out = {}
    for ub in np.unique(bval).astype(int):
        idx = np.where(bval == ub)[0]
        ref_shell = reference[..., idx]
        pred_shell = prediction[..., idx]
        residual = pred_shell[mask] - ref_shell[mask]
        mse = float(np.mean(residual ** 2))
        denom = np.sqrt(np.mean(ref_shell[mask] ** 2)) + 1e-12

        ref_mean = np.mean(ref_shell, axis=-1)
        pred_mean = np.mean(pred_shell, axis=-1)
        ref_v = ref_mean[mask]
        mean = ref_v.mean()
        std = ref_v.std()
        ref_norm = ((ref_mean - mean) / (std + 1e-12) + 3) / 6 * mask
        pred_norm = ((pred_mean - mean) / (std + 1e-12) + 3) / 6 * mask

        out[str(ub)] = {
            "MSE": round(mse, 8),
            "NRMSE": round(float(np.sqrt(mse) / denom), 6),
            "residual": array_stats(residual),
            "SSIM": round(float(ssim(ref_norm, pred_norm, data_range=1)), 6),
        }
    return out


def param_map_stats(map_dir, mask, physics):
    mask = np.asarray(mask).astype(bool)
    out = {}
    names = {
        "DTI": ["S0", "L1", "MD", "RD", "FA"],
        "DKI": ["S0", "L1", "MD", "RD", "FA", "MK", "AK", "RK"],
    }
    for name in names.get(physics, []):
        paths = glob(os.path.join(map_dir, f"{name}.nii.gz"))
        if not paths:
            continue

        metric = nib.load(paths[0]).get_fdata()[mask]
        if name in ["L1", "MD", "RD"]:
            metric = metric * 1000
        out[name] = array_stats(metric)
    return out
