"""Feature extraction for one prepared diffusion-MRI acquisition."""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
from pathlib import Path

import nibabel as nib
import numpy as np

from toolbox.utility.stat_feature import array_stats


def _one(data_dir: Path, pattern: str) -> Path:
    matches = sorted(data_dir.glob(pattern))
    if len(matches) != 1:
        raise ValueError(
            f"Expected exactly one {pattern!r} in {data_dir}, found {len(matches)}."
        )
    return matches[0]


def _load_bvec(path: Path, n_gradients: int) -> np.ndarray:
    bvec = np.asarray(np.loadtxt(path), dtype=float)
    if bvec.ndim != 2:
        raise ValueError(f"b-vectors must be a 2-D matrix: {path}")
    if bvec.shape == (3, n_gradients):
        bvec = bvec.T
    if bvec.shape != (n_gradients, 3):
        raise ValueError(
            f"Expected b-vectors with shape (3, {n_gradients}) or "
            f"({n_gradients}, 3), got {bvec.shape}."
        )
    return bvec


def _shell_label(value: float) -> str:
    return str(int(round(float(value))))


def _estimate_snr(
    dwi_path: Path,
    brain_mask_path: Path,
    tissue_mask: np.ndarray,
    bvals: np.ndarray,
    dwidenoise: Path,
) -> float:
    with tempfile.TemporaryDirectory(prefix="microagent-snr-") as temporary:
        temporary = Path(temporary)
        denoised_path = temporary / "denoised.nii.gz"
        noise_path = temporary / "noise.nii.gz"
        subprocess.run(
            [
                str(dwidenoise),
                "-mask",
                str(brain_mask_path),
                "-noise",
                str(noise_path),
                str(dwi_path),
                str(denoised_path),
                "-force",
            ],
            check=True,
        )
        denoised = nib.load(denoised_path).get_fdata(dtype=np.float32)
        noise = nib.load(noise_path).get_fdata(dtype=np.float32)

    b0 = np.mean(denoised[..., bvals < 50], axis=-1)
    valid = tissue_mask & np.isfinite(b0) & np.isfinite(noise) & (noise > 0)
    if not np.any(valid):
        raise ValueError("Could not estimate SNR: the denoising noise map is empty in-mask.")
    return round(float(np.mean(b0[valid] / noise[valid])), 2)


def extract_dwi_features(
    data_dir: str | os.PathLike,
    mrtrix3_bin: str | os.PathLike,
    field_strength: float | None = None,
    scanner_vendor: str | None = None,
) -> Path:
    """Extract the feature JSON consumed by the Architect.

    The input directory must already contain one DWI, b-value table, b-vector
    table, brain mask, tissue/weight mask, and white-matter mask. MRtrix3 is
    used only for the SNR estimate; temporary denoising files are discarded.
    """
    data_dir = Path(data_dir).expanduser().resolve()
    mrtrix3_bin = Path(mrtrix3_bin).expanduser().resolve()
    if not data_dir.is_dir():
        raise FileNotFoundError(f"DWI directory not found: {data_dir}")

    dwidenoise = mrtrix3_bin / "dwidenoise"
    if not dwidenoise.is_file() or not os.access(dwidenoise, os.X_OK):
        raise FileNotFoundError(
            f"MRtrix3 executable not found or not executable: {dwidenoise}"
        )

    dwi_path = _one(data_dir, "*_diff.nii.gz")
    bval_path = _one(data_dir, "*_diff.bval")
    bvec_path = _one(data_dir, "*_diff.bvec")
    brain_mask_path = _one(data_dir, "*_diff_mask.nii.gz")
    tissue_mask_path = _one(data_dir, "*_weight_mask.nii.gz")
    wm_mask_path = _one(data_dir, "*_evaluate_mask_WM.nii.gz")

    dwi_image = nib.load(dwi_path)
    dwi = dwi_image.get_fdata(dtype=np.float32)
    if dwi.ndim != 4:
        raise ValueError(f"DWI must be 4-D, got shape {dwi.shape}: {dwi_path}")
    bvals = np.asarray(np.loadtxt(bval_path), dtype=float).reshape(-1)
    bvec = _load_bvec(bvec_path, len(bvals))
    if dwi.shape[-1] != len(bvals):
        raise ValueError(
            f"DWI volume count ({dwi.shape[-1]}) does not match b-values "
            f"({len(bvals)})."
        )

    brain_mask = nib.load(brain_mask_path).get_fdata().astype(bool)
    tissue_mask = nib.load(tissue_mask_path).get_fdata().astype(bool)
    wm_mask = nib.load(wm_mask_path).get_fdata().astype(bool)
    for label, mask in (
        ("brain", brain_mask),
        ("tissue", tissue_mask),
        ("white-matter", wm_mask),
    ):
        if mask.shape != dwi.shape[:3]:
            raise ValueError(
                f"{label} mask shape {mask.shape} does not match DWI {dwi.shape[:3]}."
            )
    for label, mask, path in (
        ("Brain", brain_mask, brain_mask_path),
        ("Weight", tissue_mask, tissue_mask_path),
        ("White-matter", wm_mask, wm_mask_path),
    ):
        if not np.any(mask):
            raise ValueError(f"{label} mask is empty: {path}")

    rounded_bvals = np.rint(bvals).astype(int)
    unique_bvals = np.sort(np.unique(rounded_bvals))
    gradient = {}
    for shell in unique_bvals:
        shell_indices = np.where(rounded_bvals == shell)[0]
        gradient[f"num_b{shell}"] = int(len(shell_indices))
        if shell != 0:
            shell_vectors = bvec[shell_indices]
            gx, gy, gz = shell_vectors.T
            design = np.stack(
                [
                    shell * gx * gx,
                    shell * gy * gy,
                    shell * gz * gz,
                    2 * shell * gx * gy,
                    2 * shell * gx * gz,
                    2 * shell * gy * gz,
                ],
                axis=1,
            )
            gradient[f"cond_b{shell}"] = float(np.linalg.cond(design))
    gradient["num_shells"] = int(np.count_nonzero(unique_bvals))

    coordinates = np.argwhere(tissue_mask)
    lower = coordinates.min(axis=0)
    upper = coordinates.max(axis=0) + 1
    voxel_size = [float(value) for value in dwi_image.header.get_zooms()[:3]]
    spatial = {
        "voxel_size (mm)": voxel_size,
        "patch_shape": [int(value) for value in upper - lower],
        "brain_size (mm3)": float(np.sum(tissue_mask) * np.prod(voxel_size)),
    }

    signal = {
        "SNR": _estimate_snr(
            dwi_path, brain_mask_path, tissue_mask, rounded_bvals, dwidenoise
        )
    }
    b0_indices = np.where(rounded_bvals < 50)[0]
    if not len(b0_indices):
        raise ValueError("At least one b0 volume (b < 50 s/mm²) is required.")
    b0_signal = dwi[..., b0_indices]
    signal["b0"] = array_stats(b0_signal[tissue_mask])
    for shell in unique_bvals:
        if shell == 0:
            continue
        shell_signal = dwi[..., rounded_bvals == shell]
        signal[f"b{_shell_label(shell)}"] = array_stats(shell_signal[tissue_mask])

    wm_mask = wm_mask & tissue_mask
    gm_mask = tissue_mask & ~wm_mask
    if not np.any(wm_mask):
        raise ValueError(
            "White-matter and weight masks do not contain any common voxels."
        )
    if not np.any(gm_mask):
        raise ValueError(
            "The weight mask must contain at least one non-white-matter voxel."
        )
    tissue = {
        "WM_fraction": float(np.mean(wm_mask[tissue_mask])),
        "b0_WM": array_stats(b0_signal[wm_mask]),
        "b0_GM": array_stats(b0_signal[gm_mask]),
    }
    tissue["b0_WM_to_GM_ratio"] = (
        float(tissue["b0_WM"]["mean"] / (tissue["b0_GM"]["mean"] + 1e-12))
        if tissue["b0_WM"]["mean"] is not None and tissue["b0_GM"]["mean"] is not None
        else None
    )
    for shell in unique_bvals:
        if shell == 0:
            continue
        shell_signal = dwi[..., rounded_bvals == shell]
        label = f"b{_shell_label(shell)}"
        tissue[f"{label}_WM"] = array_stats(shell_signal[wm_mask])
        tissue[f"{label}_GM"] = array_stats(shell_signal[gm_mask])
        wm_mean = tissue[f"{label}_WM"]["mean"]
        gm_mean = tissue[f"{label}_GM"]["mean"]
        tissue[f"{label}_WM_to_GM_ratio"] = (
            float(wm_mean / (gm_mean + 1e-12))
            if wm_mean is not None and gm_mean is not None
            else None
        )

    features = {
        "gradient": gradient,
        "spatial": spatial,
        "signal": signal,
        "tissue": tissue,
        "other": {
            "field_strength": field_strength,
            "scanner_vendor": scanner_vendor,
        },
    }
    subject = dwi_path.name.removesuffix("_diff.nii.gz")
    output_path = data_dir / f"{subject}_features.json"
    temporary_path = output_path.with_suffix(".json.tmp")
    temporary_path.write_text(
        json.dumps(features, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary_path, output_path)
    return output_path
