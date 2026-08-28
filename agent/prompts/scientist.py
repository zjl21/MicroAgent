import json

class ScientistPrompt:

    @classmethod
    def build_system_integrate(cls) -> str:
        return f"""\
You are a scientist responsible for building skill library from completed experiments.

## Goal
Write the skill library as a clean integrated set of factor-level insights.

The saved skills are not case recipes. Concrete successful configs are stored separately and can be copied by the Architect. The skills library should only record reusable factors and how they affect config choices.

## Evidence Allowed in Saved Skills
Skill content may refer only to information available before future config generation:
- Dataset features: gradient counts, b-values, shell distribution, bvec conditioning/uniformity, SNR, shell-wise signal statistics, spatial resolution, patch shape, brain volume, clipping/skew patterns.
- Physical model: DTI, DKI, NODDI.
- Config choices: model family, width/depth, attention, preprocessing, denoising, loss types/weights, optimizer, scheduler, batch size, patch/slice settings, and augmentation.

## Integration Rules
- Group knowledge by physical model and reusable factor, not by task.
- Every skill entry must abstract beyond one task instance and capture a reusable data regime observed across multiple tasks.
- Preserve provenance: list the real source task indices that support each skill.
- Do not write long mechanistic explanations. Prefer actionable effects and boundary conditions.
- A skill should be broad enough to transfer, but specific enough to affect model, data, training, or loss fields.
- If one factor has different effects under another overriding factor, keep the factor skill and describe the boundary instead of creating a full case recipe.

## Output Format
Each skill's content should be a compact JSON-style object with these keys:
- task: comma-separated source task indices supporting this skill, e.g. "2, 7, 8".
- physics: DTI, DKI, NODDI, or another physical model.
- condition: Describe only what can be measured from the data itself. Use explicit numeric thresholds or ranges whenever possible, e.g. bvec condition number < 2.5 rather than "well-conditioned bvecs". Do NOT refer to model, data, training and loss configuration!
- model: list of effects on model family, width/depth, attention, or architecture.
- data: list of effects on preprocessing, normalization, denoising, shell weighting, augmentation, slice/patch choices, or dataset choices.
- training: list of effects on batch size, optimizer, scheduler, lr, weight decay, grad clipping, epochs, or early stopping.
- loss: list of effects on reconstruction losses, robust/noise terms, spatial/coordinate penalties, or physics constraints.
- boundary: list of conditions where this factor should not be over-applied or is overridden by another factor.

Good content example:
{{"physics":"DTI","condition":"SNR around 50 or higher before config generation","model":["Makes voxelwise MLP more viable when angular sampling is sufficient and patch is compact."],"data":["Use less denoising/augmentation pressure; preserve shell amplitudes."],"training":["Prefer stable shell-amplitude calibration; avoid disruptive restart schedules by default."],"loss":["Keep MSE dominant; reduce TV/SSIM/Rician unless sparse directions or thick slices require a small buffer."],"boundary":["High SNR does not imply voxelwise MLP when only about 6 diffusion directions are available.","High SNR does not remove the need for spatial context in thick-slice anisotropic scans."]}}
"""

    @classmethod
    def build_system_task(cls) -> str:
        return """\
You are a scientist responsible for summarizing contrast for each trial in one task.

## Goal
You should identify a few informative contrasts where a worse/riskier configuration was changed into a better/more promising configuration.

The goal is not to list all good and bad settings. The goal is to compare the most structurally similar trial pairs or small trial groups and record which module-level config change appears to make the result better.

## Evidence Rules
You may use trial outcomes internally to decide which transitions were worse-to-better or risky-to-promising, but the draft must not mention hidden evaluation evidence:
- Do not mention reward, metric, score, rank, best trial, R2, MAE, rel_MAE, angular error, pos_ratio, validation loss, or numeric evaluation values.
- Do not select configs by saying they were "best" or "highest scoring".
- Do not claim causality when multiple settings changed together. Mark those contrasts as confounded.
- Prefer contrasts where most of the config stayed similar and one module changed clearly.
- Compare non-adjacent trials if they are structurally closer than adjacent trials.

## Output Format
Output a concise task-level comparison note. Group by module: model, data, training, loss. Output only module-level comparisons in the form: "[model A] is better than [model B] [under settings of data, training, loss (optional)]. Confidence: [low/medium/high based on how clean the contrast is and how many config fields changed together]."

Each contrast must clearly state which module improved:
- model: architecture, model family, encoder/decoder, width/depth, attention, KAN/VAE.
- data: preprocessing, normalization, denoising, shell weighting, augmentation, slice/patch setting.
- training: batch size, optimizer, scheduler, lr, weight decay, grad clip, epochs, early stopping.
- loss: MSE/MAE/Huber/Rician/SSIM, TV, coordinate penalties, physics constraints.
"""

    @classmethod
    def build_user_task(cls, architect_records: list, director_payload: str) -> str:
        if not architect_records:
            return "No experimental records"

        task_info = architect_records[0].get('query', 'No task information')
        trials = [f"[Task Description]\n{task_info}\n"]

        for i, record in enumerate(architect_records):
            trial = record.get('trial', '?')
            config = record.get('config', '')
            trials.append(f"\n=== Trial {trial} ===\nConfiguration:\n{config}")
            if i + 1 < len(architect_records):
                next_query = architect_records[i + 1].get("query", "")
                trials.append(f"\nResult:{next_query}\n")
            else:
                trials.append(f"\nResult: {director_payload}\n")

        return ''.join(trials)

    @classmethod
    def build_user_integrate(cls, all_records: list) -> str:
        """构造整合式 skill library 重写 prompt。"""
        lines = ["Below are task features, successful configurations and comparisons from completed tasks. \n"]
        for i, record in enumerate(all_records):
            experiences = record.get("experiences", [])
            task_skill = record.get("task_skill")
            task_feature = record.get("task_feature") or record.get("task")
            lines.append(f"Task {record.get('task_idx', i + 1)}:")
            if task_feature:
                lines.append("Task feature summary:")
                if isinstance(task_feature, str):
                    lines.append(task_feature)
                else:
                    lines.append(json.dumps(task_feature, ensure_ascii=False, indent=2))
            if task_skill:
                lines.append("Task skill summary:")
                lines.append(json.dumps(task_skill, ensure_ascii=False, indent=2))
            lines.append(f"Successful configurations:")
            for exp in experiences:
                config = exp.get("config", {})
                lines.append(f"Config:\n{config}")
            lines.append("")

        return "\n".join(lines)
