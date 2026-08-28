import json
import math
import os
from glob import glob

import yaml

from agent import runtime
from agent.tools.gpu_holder import DEFAULT_PROCESS_MIB

runtime.configure_library_path()


class ArchitectPrompt:
    """Build Architect prompts by combining generic agent rules with a library-local task/task.py."""

    AGENT_INSTRUCTIONS_TEMPLATE = """\
Your task is to generate a complete training config file,
which is used to define the model structure and training strategy for this project.
Please make full use of the allocated GPU memory. Each task process is limited to {process_mib}MiB of VRAM, not a full A800 card. Please use a reasonable batch_size and network size within this memory budget.

Before generating the configuration file, you must first think and plan the parameters you want to generate.
Please refer to the current task library's task/config_reference.yaml for the output of the config.
Your output must strictly contain two parts:
1. `<thought>` tag: Your thought process when building this config. If it is a modification triggered by the Director's failure diagnosis, please explain how you inferred the root cause from the diagnostic phenomenon and autonomously decided on the solution; if this is the first time, explain your design philosophy.
2. `<config>` tag: Contains only pure YAML content. Please do not add extra ```yaml code blocks, just write yaml directly.

For example, your output format should strictly be as follows:
<thought>
The director pointed out that the current learning rate is too high, causing oscillations, and expects to add TV regularization.
Here I lowered the initial learning rate from 1e-3 to 1e-4.
In addition...
</thought>
<config>
training:
epochs: 100
learning_rate: 0.0001
...
</config>

Failure is normal, and every failure accumulates experience.
Do not be too conservative; dare to try edge cases."""

    NOT_CONFIG = """\
Did not find the <config> tag or the format is incorrect. Please strictly use the <thought> and <config> tags for output!"""

    NOT_VALID = """\
If you strongly need this feature, you must explain to humans why you need this out-of-bounds field in the `<issue>...</issue>` tag of your next reply. But within `<config>`, please obey the rules of the currently available operator dictionary and forcibly use plain alternatives! Please output again."""

    NOT_YAML = """\
Please do not add extra markdown markers. Ensure that the <config> is in pure, valid YAML format."""

    @classmethod
    def _task_attr(cls, name: str, default=None):
        return getattr(runtime.task_prompt(), name, default)

    @classmethod
    def _call_task_func(cls, name: str, *args):
        fn = cls._task_attr(name)
        if fn is None:
            return None
        if isinstance(fn, classmethod):
            fn = fn.__func__
        return fn(cls, *args)

    @classmethod
    def _build_task_context(cls) -> str:
        module = runtime.task_prompt()
        identity = getattr(module, "identity", "").strip()
        project_description = getattr(module, "project_description", "").strip()
        sections = getattr(module, "task_context_sections", (
            ("Core Training Definition", "core_training_definition"),
            ("Task Nature", "task_nature"),
        ))

        parts = [identity, project_description]
        for title, attr_name in sections:
            content = getattr(module, attr_name, "").strip()
            if content:
                parts.append(
                    f"------------------------------------------------------------\n"
                    f"{title}\n"
                    f"------------------------------------------------------------\n\n"
                    f"{content}"
                )
        return "\n\n".join(part for part in parts if part)

    @classmethod
    def _build_operator_dictionary(cls, reference_content: str) -> str:
        return f"""
        ==============================
        Below is the [Available Operator Dictionary] of all parameters and their formats supported by the system.
        Note: This is a rough reference listing some possible options, not an unchangeable fixed template!
        === Operator Dictionary ===
        ```yaml
        {reference_content}
        ```
        =============================="""

    @classmethod
    def _experiment_process_mib(cls, exp_dir: str) -> int:
        if exp_dir is None:
            return DEFAULT_PROCESS_MIB
        cfg_path = os.path.join(exp_dir, "agent_config.json")
        if os.path.exists(cfg_path):
            with open(cfg_path, "r", encoding="utf-8") as f:
                return int(json.load(f).get("process_mib", DEFAULT_PROCESS_MIB))
        return DEFAULT_PROCESS_MIB

    @classmethod
    def _agent_instructions(cls, exp_dir: str) -> str:
        return cls.AGENT_INSTRUCTIONS_TEMPLATE.format(
            process_mib=cls._experiment_process_mib(exp_dir)
        )

    @classmethod
    def _format_skill_context(cls, skill) -> str:
        if not skill:
            return ""

        if isinstance(skill, str):
            skills_text = skill.strip()
        else:
            skill_lines = []
            for s in skill:
                if isinstance(s, dict) and not s:
                    continue
                text = str(s.get("content", s)).strip() if isinstance(s, dict) else str(s).strip()
                if text:
                    skill_lines.append(text)
            skills_text = "\n".join(skill_lines)
        if not skills_text:
            return ""
        return f"\n\n[Skill: local interpolation rules in data-characteristic space]\n{skills_text}\n"

    @classmethod
    def _format_experience_strategy(
        cls,
        has_successful_experience: bool,
        has_skill: bool,
    ) -> str:
        if has_successful_experience and has_skill:
            return """

[How to use successful experiences and skill]
Treat successful experiences as sparse anchor points in a mapping from measurable data characteristics to configurations. Treat the skill as local rules for moving between comparable anchors or from the nearest anchor toward a target that remains inside the skill's supported region.
1. Match physics first, then compare shell/b-value sampling, direction and b0 counts, b-vector conditioning, SNR/signal statistics, voxel size/anisotropy, spatial size, and artifacts to find the nearest comparable successful anchor. Prefer anchors that bracket the current numeric characteristics when available.
2. Start from the nearest anchor configuration. Apply only skill entries whose condition matches the current data and use their concrete numeric ranges to adjust the affected fields toward the current data characteristics.
3. Preserve fields from the nearest anchor when no applicable skill says they should change. Do not average unrelated successful configurations.
4. Interpolate numeric fields only inside the skill's supported data range. For categorical fields such as model family, preprocessing, denoising, optimizer, or scheduler, switch only when the skill gives a matching threshold or regime boundary.
5. Never extrapolate a skill beyond its boundary. If the current data lies outside all supported regions, fall back to the nearest comparable anchor and make only changes required by hard constraints, operator validity, and memory feasibility.
"""
        if has_successful_experience:
            return """

[How to use successful experiences]
Treat successful experiences as sparse anchor points in a mapping from measurable data characteristics to configurations. Because no interpolation skill is available, use nearest-neighbor selection:
1. Match physics first, then compare shell/b-value sampling, direction and b0 counts, b-vector conditioning, SNR/signal statistics, voxel size/anisotropy, spatial size, and artifacts.
2. Select the single nearest comparable successful anchor and use its configuration as the baseline. Do not average numeric or categorical fields across different anchors.
3. Change the baseline only when required by the current hard constraints, operator validity, or memory feasibility; do not invent an unsupported trend between anchors.
"""
        if has_skill:
            return """

[How to use skill without raw successful anchors]
Without successful anchor configurations, true interpolation is impossible. Treat the skill as a compressed piecewise mapping from data characteristics to absolute configuration ranges:
1. Match physics, then select the most specific skill conditions satisfied by the current data. A condition combining more relevant measured characteristics takes precedence over a broader overlapping condition.
2. Instantiate affected fields directly from the skill's absolute numeric ranges and categorical transition conditions. Do not interpret relative words as usable values and do not fabricate a missing anchor.
3. When multiple matching skills affect disjoint modules or fields, combine them. When they conflict on the same field, prefer the more specific condition; if specificity is equal and the values conflict, do not average them and use current-task reasoning instead.
4. Fill fields not covered by skill from the operator dictionary, hard constraints, memory feasibility, and current-task reasoning.
5. Do not apply a rule outside its stated boundary. If no skill condition covers the current data, fall back to ordinary Architect reasoning rather than extrapolating.
"""
        return ""

    @classmethod
    def _collect_skill_task_indices(cls, skill) -> set:
        task_indices = set()
        if not isinstance(skill, list):
            return task_indices
        for s in skill or []:
            if not isinstance(s, dict):
                continue
            task_str = s.get("task", "")
            for t in task_str.split(","):
                t = t.strip()
                if not t:
                    continue
                try:
                    task_indices.add(int(t))
                except ValueError:
                    pass
        return task_indices

    @classmethod
    def _read_task_inputs(cls, exp_dir, task_idx):
        fpCurriculum = os.path.join(exp_dir, "curriculum.json")
        if not os.path.isfile(fpCurriculum):
            return None, None, None
        try:
            with open(fpCurriculum, "r", encoding="utf-8") as f:
                curriculum = json.load(f)
        except (OSError, json.JSONDecodeError):
            return None, None, None
        task_info = next((item for item in curriculum if item["idx"] == task_idx), None)
        if not task_info:
            return None, None, None
        data_dir = os.path.join(exp_dir, f"task_{task_idx}", "data")
        feature_paths = glob(os.path.join(data_dir, "*_features.json"))
        if not feature_paths:
            return None, None, None
        fpData = feature_paths[0]
        with open(fpData, "r", encoding="utf-8") as f:
            dataset_features = json.load(f)
        return dataset_features, task_info["model"], data_dir

    @classmethod
    def build_system(
        cls,
        reference_content: str,
        exp_dir: str,
        max_trials: int = 20,
        skill: list = None,
        use_successful_experience: bool = False,
        model: str = None,
        target_features_path: str = None,
        required_model_family: str = None,
    ) -> str:
        system = "\n\n".join([
            cls._build_task_context(),
            cls._agent_instructions(exp_dir),
            f"You have a total of {max_trials} rounds of time for debugging.\n",
            cls._build_operator_dictionary(reference_content),
        ])

        successful_context = (
            cls._format_successful_experiences(
                exp_dir,
                model=model,
                target_features=cls._load_features(target_features_path),
                required_model_family=required_model_family,
            )
            if use_successful_experience
            else ""
        )
        skill_context = cls._format_skill_context(skill)
        system += cls._format_experience_strategy(
            bool(successful_context), bool(skill_context)
        )
        system += successful_context
        system += skill_context
        task_indices = cls._collect_skill_task_indices(skill)
        if task_indices and not successful_context:
            system += cls._format_task_descriptions(exp_dir, task_indices)
        return system

    @classmethod
    def build_system_test(
        cls,
        reference_content: str,
        exp_dir: str,
        skill: list = None,
        use_successful_experience: bool = False,
        model: str = None,
        target_features_path: str = None,
        required_model_family: str = None,
    ) -> str:
        parts = [cls._build_task_context(), cls._agent_instructions(exp_dir)]
        parts.extend([
            "You have only one chance to test on the current task. Please choose the most suitable configuration based on past experience and the above operator dictionary to test on the current task.",
            cls._build_operator_dictionary(reference_content),
        ])
        system = "\n\n".join(parts)

        successful_context = (
            cls._format_successful_experiences(
                exp_dir,
                model=model,
                target_features=cls._load_features(target_features_path),
                required_model_family=required_model_family,
            )
            if use_successful_experience
            else ""
        )
        skill_context = cls._format_skill_context(skill)
        system += cls._format_experience_strategy(bool(successful_context), bool(skill_context))
        system += successful_context
        system += skill_context

        task_indices = cls._collect_skill_task_indices(skill)
        if task_indices and not successful_context:
            system += cls._format_task_descriptions(exp_dir, task_indices)
        return system

    @classmethod
    def build_user(cls, task_dir: str, task_idx: int, hard_requirements, past_records: list = [], summary: str = "", diagnosis: str = "") -> str:
        exp_dir = os.path.dirname(task_dir)
        if past_records == []:
            user_prompt = cls.build_task_description(exp_dir, int(task_idx)) + "Please design the initial configuration."
        else:
            user_prompt = f"Dataset and Requirements:\n{cls.build_task_description(exp_dir, int(task_idx))}\n\n"
            history_ctx = "[Past Design History (Please be sure to learn from these lessons to avoid repeating ineffective designs)]\n"
            for i, record in enumerate(past_records):
                history_ctx += f"--- History Round {i+1} ---\n"
                history_ctx += f"Configuration generated at the time:\n```yaml\n{record.get('config', '')}\n```\n"
                prev_payload = past_records[i+1].get("query", "") if i+1 < len(past_records) else summary
                history_ctx += f"Summary received at the time:\n{prev_payload}\n\n"
            history_ctx += f"The possible issues for this round are:\n{diagnosis}\n\nPlease design a new configuration plan based on the above historical experience."
            user_prompt = f"{user_prompt}\n\n{history_ctx}\n\nPlease autonomously analyze the problem and design a solution based on the above diagnosis, and output a new configuration plan combining the operator dictionary and historical lessons."

        return cls._append_hard_requirements(user_prompt, hard_requirements)

    @classmethod
    def build_user_test(cls, fpFeat, model, hard_requirements) -> str:
        with open(fpFeat, "r", encoding="utf-8") as f:
            dataset_features = json.load(f)
        user_prompt = cls._describe_current_task(dataset_features, model, data_dir=None)
        user_prompt += "Please design the configuration for testing on the current task, based on the evidence and operator dictionary provided above."
        return cls._append_hard_requirements(user_prompt, hard_requirements)

    @classmethod
    def build_task_description(cls, exp_dir, task_idx) -> str:
        dataset_features, model, data_dir = cls._read_task_inputs(exp_dir, int(task_idx))
        if dataset_features is None:
            return "None"
        return cls._describe_current_task(dataset_features, model, data_dir)

    @classmethod
    def _describe_current_task(cls, dataset_features, model, data_dir=None) -> str:
        return cls._call_task_func("build_task_description", dataset_features, model)

    @classmethod
    def _append_hard_requirements(cls, user_prompt: str, hard_requirements) -> str:
        if not hard_requirements:
            return user_prompt
        if isinstance(hard_requirements, (dict, str)):
            hard_requirements = [hard_requirements]

        lines = []
        for requirement in hard_requirements:
            if isinstance(requirement, dict):
                lines.extend(
                    f'- Please be sure to set `{k}` to: {json.dumps(v, ensure_ascii=False)}.'
                    for k, v in requirement.items()
                )
            elif isinstance(requirement, str):
                text = requirement.strip()
                if text:
                    lines.append(f"- {text}")
            else:
                raise TypeError(f"Unsupported hard requirement type: {type(requirement).__name__}")
        if not lines:
            return user_prompt
        lines = "\n".join(lines)
        return f"{user_prompt}\n\n[System hard routing restrictions, cannot be violated]:\n{lines}"

    @classmethod
    def _format_task_descriptions(cls, exp_dir: str, task_indices: set) -> str:
        text = "\nThe descriptions of the tasks involved in the experience are as follows:\n"
        found = False
        for task_idx in sorted(task_indices):
            task_desc = cls.build_task_description(exp_dir, task_idx)
            if task_desc and task_desc != "None":
                text += f"\n[Task {task_idx}]\n{task_desc}\n"
                found = True
        return text if found else ""

    @staticmethod
    def _load_features(path: str):
        if not path or not os.path.isfile(path):
            return None
        try:
            with open(path, "r", encoding="utf-8") as handle:
                value = json.load(handle)
            return value if isinstance(value, dict) else None
        except (OSError, json.JSONDecodeError):
            return None

    @staticmethod
    def _extract_experience_features(task_metadata: str):
        marker = "Dataset Features:"
        if not isinstance(task_metadata, str) or marker not in task_metadata:
            return None
        tail = task_metadata.split(marker, 1)[1].lstrip()
        try:
            value, _ = json.JSONDecoder().raw_decode(tail)
            return value if isinstance(value, dict) else None
        except json.JSONDecodeError:
            return None

    @staticmethod
    def _feature_distance(target: dict, candidate: dict) -> float:
        """Scale-free distance used only to retrieve nearby experience anchors."""
        if not target or not candidate:
            return math.inf
        deltas = []

        def add(left, right):
            if isinstance(left, (int, float)) and isinstance(right, (int, float)):
                deltas.append(abs(math.log1p(abs(float(left))) - math.log1p(abs(float(right)))))
            elif isinstance(left, list) and isinstance(right, list):
                for lv, rv in zip(left, right):
                    add(lv, rv)

        target_gradient = target.get("gradient", {})
        candidate_gradient = candidate.get("gradient", {})
        for key in sorted(set(target_gradient) & set(candidate_gradient)):
            if key.startswith(("num_", "cond_")):
                add(target_gradient[key], candidate_gradient[key])

        target_spatial = target.get("spatial", {})
        candidate_spatial = candidate.get("spatial", {})
        for key in ("voxel_size (mm)", "patch_shape", "brain_size (mm3)"):
            add(target_spatial.get(key), candidate_spatial.get(key))

        add(target.get("signal", {}).get("SNR"), candidate.get("signal", {}).get("SNR"))
        return sum(deltas) / len(deltas) if deltas else math.inf

    @classmethod
    def _experience_matches_model(
        cls,
        config_text: str,
        model: str,
        required_model_family: str = None,
    ) -> bool:
        if not config_text:
            return False
        try:
            config = yaml.safe_load(config_text)
        except yaml.YAMLError:
            return False
        if not isinstance(config, dict):
            return False
        if model:
            physics = config.get("physics", {})
            experience_model = physics.get("name") if isinstance(physics, dict) else None
            if not (
                isinstance(experience_model, str)
                and experience_model.strip().casefold() == str(model).strip().casefold()
            ):
                return False
        if required_model_family:
            model_config = config.get("model", {})
            family = model_config.get("name") if isinstance(model_config, dict) else None
            if not (
                isinstance(family, str)
                and family.strip().casefold()
                == str(required_model_family).strip().casefold()
            ):
                return False
        return True

    @classmethod
    def _format_successful_experiences(
        cls,
        exp_dir: str,
        model: str = None,
        target_features: dict = None,
        required_model_family: str = None,
        max_tasks: int = 3,
        max_configs_per_task: int = 3,
    ) -> str:
        text = ""
        success_dir = os.path.join(exp_dir, "success")
        experience_files = glob(os.path.join(success_dir, "task_*.json"))
        anchors = []
        for experience_file in experience_files:
            task_idx = os.path.basename(experience_file).split("_")[1].split(".")[0]
            with open(experience_file, "r", encoding="utf-8") as f:
                experience = json.load(f)
            exp_list = experience.get("experiences", [])
            if isinstance(exp_list, list) and exp_list:
                matching_items = [
                    item for item in exp_list
                    if isinstance(item, dict)
                    and cls._experience_matches_model(
                        item.get("config", ""), model, required_model_family
                    )
                ]
            else:
                config = experience.get("config", "")
                matching_items = (
                    [experience]
                    if cls._experience_matches_model(
                        config, model, required_model_family
                    )
                    else []
                )

            if not matching_items:
                continue

            matching_items = sorted(
                matching_items,
                key=lambda item: (
                    float(item.get("score"))
                    if isinstance(item.get("score"), (int, float))
                    else math.inf
                ),
            )[:max_configs_per_task]

            task_metadata = experience.get("task", "")
            distance = cls._feature_distance(
                target_features,
                cls._extract_experience_features(task_metadata),
            )
            anchors.append((distance, int(task_idx), task_metadata, matching_items, bool(exp_list)))

        anchors.sort(key=lambda item: (item[0], item[1]))
        if target_features:
            anchors = anchors[:max_tasks]

        for distance, task_idx, task_metadata, matching_items, has_list in anchors:
            if task_metadata:
                distance_text = f" (retrieval distance: {distance:.4f})" if math.isfinite(distance) else ""
                text += f"\n\nMetadata for successful task {task_idx}{distance_text}:\n{task_metadata}\n"
            if has_list:
                text += f"\n\nSuccessful experiences from task {task_idx}:\n"
                for i, item in enumerate(matching_items, 1):
                    cfg = item.get("config", "")
                    score = item.get("score", None)
                    score_text = f" (score: {score})" if score is not None else ""
                    text += f"\n[Candidate {i}{score_text}]\n{cfg}\n"
            else:
                text += f"\n\nSuccessful experience from task {task_idx}:\n{matching_items[0].get('config', '')}\n"
        return text
