import json
import os

from agent.runtime import configure_library_path, reward_interpretation_path

configure_library_path()

class DirectorPrompt:

    IDENTITY = """\
You are a top-tier Deep Learning Algorithm Director.
Your task is to translate the input json file into a semantic report and guide the architect the next step.
---"""

    # ── 情境说明（二选一）────────────────────────────────────────────────

    WITH_GRAD = """\
## Current Situation

There are issues with the gradients in this training run. The system has completed the detection and attached the report in the prompt (including the norm, mean, and max statistics of the gradients for each layer).

**Your Task**: Objectively describe the location and phenomena of the problem based on the gradient report (e.g., which layer exploded, which layer vanished). Do not provide specific modification suggestions; let the Architect decide how to adjust it on their own.

---

## Gradient Report Interpretation Guide

### Gradient Disconnection / Frozen Layers
- The norm of a parameter layer is `null` or `0.0`: The layer did not participate in backpropagation, became detached from the computation graph, or was incorrectly frozen.
- Troubleshooting direction: Check if the layer has `requires_grad=False`, or if an operation in the physics layer disconnected the computation graph (e.g., numpy conversion, in-place operations).

### Gradient Explosion
- max or mean shows massive values (> 1e4) or contains NaN/Inf.
- Modification suggestions: Lower the learning rate, introduce or reduce `grad_clip`, or check if the numerical scale of the physics layer is reasonable.

### Gradient Vanishing
- norm is extremely small (< 1e-6), approaching zero.
- Modification suggestions: Adjust the loss function weights to make the gradient signal stronger, change the activation function (e.g., ReLU → LeakyReLU), or check the normalization method of input features.

### Gradient Imbalance Between Physics Layer and Backbone Network
- Compare the gradient norm of the physics penalty layer (the layer corresponding to the PINN loss) with the backbone network (MLP/CNN backbone).
- If the difference is more than two orders of magnitude: It indicates that the multi-task weight allocation is unreasonable, and the gradient is dominated by one term. It is recommended to adjust the weight ratio of each loss component so that the gradient magnitudes of both parts are comparable.

---"""

    NO_GRAD_ISSUE = """\
## Current Situation

The gradients for this round of training are normal. The system has automatically used the best checkpoint to complete whole-brain inference. Below are the training logs and the parameter map quality evaluation report for this training run (metrics such as R²/RMSE/Bias/Angular Error).

**Your Task**: Objectively describe the shortcomings of each metric and their possible causes. Do not provide specific modification suggestions; let the Architect decide how to adjust it on their own.

---

## Historical Comparison Metrics

Each metric includes historical comparison information to help you judge whether the current training is improving or regressing:

- **value**: The actual metric value for this trial
- **better**: Indicates whether "higher" or "lower" values are better for this metric
- **vs_best**: Current value / historical best value
- **vs_mean**: Current value / historical mean value
- **vs_last**: (Current - Last) / Last (percentage change from previous trial)
- **is_best**: Boolean indicating if this is the best value ever achieved
- **norm_value**: Normalized value in range [0, 1] across all trials (0=worst, 1=best). This allows direct comparison across different metrics.

---

{task_reward_interpretation}"""

    # ── 输出格式（始终附加）────────────────────────────────────────────────

    OUTPUT_FORMAT = """\
## Output Format

You [MUST] output strictly using the following XML tag format:

```
<summary>A refined review and summary of this experiment (how the overall result was, which metrics were abnormal). Please state here all the **phenomena and numerical values** you noticed.</summary>
<failure_diagnosis>
Speculate on the causes of the problems in this round (e.g., which layer's gradient was abnormal, whether there was overfitting). Only speculate on possible causes, **do not mention specific numerical values**.
</failure_diagnosis>
```"""
    # ── 组装接口 ──────────────────────────────────────────────────────────

    @classmethod
    def build_system(cls, has_grad: bool = False) -> str:
        task_reward_interpretation = ""
        if not has_grad:
            path = reward_interpretation_path()
            if os.path.exists(path):
                with open(path, "r", encoding="utf-8") as f:
                    task_reward_interpretation = f.read().strip()
        situation = cls.WITH_GRAD if has_grad else cls.NO_GRAD_ISSUE.format(
            task_reward_interpretation=task_reward_interpretation
        )
        return f"{cls.IDENTITY}\n\n{situation}\n\n{cls.OUTPUT_FORMAT}"

    @classmethod
    def build_user(cls, config_content: str, history: list, extra_info: str = "") -> str:
        return (
            f"[Experiment configuration for this recently completed round (config.yaml)]:\n```yaml\n{config_content}\n```\n\n"
            f"[Logs after completion of this training round]:\n{json.dumps(history, indent=2)}"
            f"{extra_info}\n\n"
            f"Please evaluate and output parsing instructions strictly in the requested XML format."
        )
