"""One-command MicroAgent DTI quantification workflow."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import yaml

from toolbox.utility.features import extract_dwi_features
from toolbox.utility.paths import set_project_root


PROJECT_ROOT = Path(__file__).resolve().parent
DTI_OUTPUTS = (
    "tensor.nii.gz", "S0.nii.gz", "FA.nii.gz", "MD.nii.gz",
    "RD.nii.gz", "value.nii.gz", "vector.nii.gz", "L1.nii.gz",
    "L2.nii.gz", "L3.nii.gz", "V1.nii.gz", "V2.nii.gz", "V3.nii.gz",
)
REQUIRED_INPUT_SUFFIXES = (
    "_diff.nii.gz",
    "_diff_mask.nii.gz",
    "_diff.bval",
    "_diff.bvec",
    "_weight_mask.nii.gz",
    "_evaluate_mask_WM.nii.gz",
)


def _resolve_config(value: str) -> Path:
    candidate = Path(value).expanduser()
    candidates = [candidate]
    if not candidate.suffix:
        candidates.extend(candidate.with_suffix(suffix) for suffix in (".yaml", ".yml"))
    for path in candidates:
        if path.is_file():
            return path.resolve()
    raise FileNotFoundError(f"Environment config not found: {value}")


def _resolve_path(value: str, label: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty path.")
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def _validate_input_files(data_dir: Path) -> None:
    resolved = {}
    problems = []
    for suffix in REQUIRED_INPUT_SUFFIXES:
        matches = sorted(data_dir.glob(f"*{suffix}"))
        if len(matches) != 1:
            problems.append(f"*{suffix}: expected 1 file, found {len(matches)}")
        else:
            resolved[suffix] = matches[0]
    if problems:
        details = "; ".join(problems)
        raise ValueError(f"Invalid DWI input directory {data_dir}: {details}")

    prefixes = {
        path.name[:-len(suffix)]
        for suffix, path in resolved.items()
    }
    if len(prefixes) != 1:
        raise ValueError(
            f"All six DWI input files in {data_dir} must share one prefix; "
            f"found {sorted(prefixes)}."
        )


def _load_environment(path: Path) -> dict:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError(f"{path} must contain one YAML object.")

    api = config.get("api")
    if not isinstance(api, dict):
        raise ValueError("env_config.api must contain provider, base_url, and model_name.")
    required_api = ("provider", "base_url", "model_name")
    missing = [key for key in required_api if not str(api.get(key, "")).strip()]
    if missing:
        raise ValueError(f"Missing API fields: {missing}")
    if str(api["provider"]).strip().lower() != "openai":
        raise ValueError("MicroAgent release supports the shared OpenAI-compatible API only.")
    if any("REPLACE_WITH" in str(api[key]) for key in required_api):
        raise ValueError("Replace the API placeholders in env_config before running.")
    if not os.environ.get("API_KEY", "").strip():
        raise ValueError("Set the API_KEY environment variable before running.")

    config["data_dir"] = str(_resolve_path(config.get("data_dir"), "data_dir"))
    config["output_dir"] = str(_resolve_path(config.get("output_dir"), "output_dir"))
    config["mrtrix3_bin"] = str(_resolve_path(config.get("mrtrix3_bin"), "mrtrix3_bin"))
    config["gpu_id"] = int(config.get("gpu_id", 0))
    config["gpu_memory_limit_mib"] = int(config.get("gpu_memory_limit_mib", 12000))
    if config["gpu_memory_limit_mib"] < 1:
        raise ValueError("gpu_memory_limit_mib must be positive.")
    hard_requirements = config.get("hard_requirements", {})
    if not isinstance(hard_requirements, (dict, list, str)):
        raise ValueError("hard_requirements must be a mapping, list, or string.")

    data_dir = Path(config["data_dir"])
    if not data_dir.is_dir():
        raise FileNotFoundError(f"DWI directory not found: {data_dir}")
    _validate_input_files(data_dir)
    mrtrix_bin = Path(config["mrtrix3_bin"])
    dwidenoise = mrtrix_bin / "dwidenoise"
    if not dwidenoise.is_file() or not os.access(dwidenoise, os.X_OK):
        raise FileNotFoundError(f"Executable dwidenoise not found in {mrtrix_bin}")
    return config


def _write_api_config(api: dict, run_dir: Path) -> Path:
    payload = {
        "provider": "openai",
        "base_url": str(api["base_url"]).strip(),
        "model_name": str(api["model_name"]).strip(),
    }
    path = run_dir / "api.json"
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


def _existing_feature_path(data_dir: str | os.PathLike) -> Path | None:
    """Return the feature file matching the directory's single DWI, if present."""
    data_dir = Path(data_dir)
    dwi_paths = sorted(data_dir.glob("*_diff.nii.gz"))
    if len(dwi_paths) != 1:
        raise ValueError(
            f"Expected exactly one '*_diff.nii.gz' in {data_dir}, "
            f"found {len(dwi_paths)}."
        )
    subject = dwi_paths[0].name.removesuffix("_diff.nii.gz")
    feature_path = data_dir / f"{subject}_features.json"
    return feature_path if feature_path.is_file() else None


def _build_training_config(config: dict, run_dir: Path) -> Path:
    set_project_root(str(PROJECT_ROOT))
    os.environ.setdefault("AGENT_LIBRARY", "quantification")
    from agent.runtime import configure_library_path, task_prompt

    configure_library_path()
    from agent.roles.architect import Architect
    from agent.tools.skill import SkillManager
    from agent.tools.utility import (
        build_task_hard_requirements,
        normalize_hard_requirements,
    )

    task = task_prompt()
    requirements = normalize_hard_requirements(task.default_hard_requirements())
    requirements.extend(normalize_hard_requirements(config.get("hard_requirements")))
    requirements = build_task_hard_requirements(
        requirements,
        {"model": "DTI"},
        config["data_dir"],
        str(config["gpu_id"]),
        task,
        config["gpu_memory_limit_mib"],
    )

    skill_dir = PROJECT_ROOT / "skill"
    skills = SkillManager(str(skill_dir)).load_text()
    architect = Architect(
        task_dir=config["data_dir"],
        hard_requirements=requirements,
        skill=skills,
        test=True,
        resume_dir=str(skill_dir),
        use_successful_experience=True,
        physics_model="DTI",
        quiet=True
    )
    architect.api_config_path = str(run_dir / "api.json")
    architect.design_test(trial_dir=str(run_dir), model="DTI")

    generated = run_dir / "config.yaml"
    if not generated.is_file():
        raise RuntimeError(f"Architect did not generate a config: {generated}")
    return generated


def _run(command: list[str]) -> None:
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate a DTI config, self-supervise, and run whole-volume inference."
    )
    parser.add_argument("--config", required=True, help="Path to env_config YAML")
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="Validate configuration and inputs without extracting features or running.",
    )
    args = parser.parse_args()

    env_path = _resolve_config(args.config)
    config = _load_environment(env_path)
    if args.preflight_only:
        print("MicroAgent preflight passed.")
        return

    mrtrix_bin = Path(config["mrtrix3_bin"])
    os.environ["MRTRIX3_BIN"] = str(mrtrix_bin)
    os.environ["PATH"] = os.pathsep.join([str(mrtrix_bin), os.environ.get("PATH", "")])
    os.environ.setdefault("AGENT_LIBRARY", "quantification")

    feature_path = _existing_feature_path(config["data_dir"])
    if feature_path is None:
        print("[1/4] Extracting DWI features...")
        feature_path = extract_dwi_features(
            config["data_dir"],
            mrtrix_bin,
            field_strength=config.get("field_strength"),
            scanner_vendor=config.get("scanner_vendor"),
        )
    else:
        print("[1/4] Reusing existing DWI features...")
    print(f"      {feature_path}")

    run_dir = Path(config["output_dir"])
    run_dir.mkdir(parents=True, exist_ok=True)
    _write_api_config(config["api"], run_dir)
    training_config = run_dir / "config.yaml"
    if not training_config.is_file():
        print("[2/4] Generating the training config with the packaged skill...")
        training_config = _build_training_config(config, run_dir)
    else:
        print("[2/4] Reusing the generated training config.")

    from agent.tools.utility import rewrite_config_gpu

    rewrite_config_gpu(
        str(training_config),
        config["gpu_id"],
        config["gpu_memory_limit_mib"],
    )

    print("[3/4] Running or resuming self-supervised training...")
    _run([sys.executable, str(PROJECT_ROOT / "train.py"), "--config", str(training_config)])

    print("[4/4] Running whole-volume DTI inference...")
    _run([sys.executable, str(PROJECT_ROOT / "infer.py"), "--config", str(training_config)])
    output_dir = run_dir / "output"
    missing = [name for name in DTI_OUTPUTS if not (output_dir / name).is_file()]
    if missing:
        raise RuntimeError(f"DTI inference completed with missing outputs: {missing}")
    print(f"MicroAgent finished. DTI maps: {output_dir}")


if __name__ == "__main__":
    main()
