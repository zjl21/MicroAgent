import json
from typing import Union
# ---------------------------------------------------------------------------
# User-editable task knowledge
# ---------------------------------------------------------------------------
# This file belongs to the task library. Users describe the task here; the
# generic Architect prompt loader in agent/prompts/architect.py reads these
# variables and functions when it builds the system/user prompts.

identity = "You are a research assistant familiar with diffusion MRI and physics-informed neural networks."

project_description = "This project is a **self-supervised** physics-constrained inverse problem."

core_training_definition = """
The network predicts physical parameter maps from diffusion MRI signals.
These parameters are passed through a fixed, untrainable physical forward model to reconstruct the observed signals.

Training does not use ground-truth parameter maps as labels. The objective combines:
1) Data consistency loss: matches reconstructed signals to observed signals.
2) Parameter regularization loss: encourages physically plausible and spatially reasonable maps.

Overall objective:
L_total = L_data + λ_reg * L_regularization
"""

task_nature = """This is a self-supervised, physics-constrained inverse problem.
The main supervision comes from signal reconstruction through the physical model.
The goal is to find parameter maps that fit the measured signals and remain physically reasonable."""

task_budget = """`training.epochs` should be below 1000 in most cases. Only when there is a clear need, it may be set in the 1000-1500 range."""

task_context_sections = (
    ("Core Training Definition", "core_training_definition"),
    ("Task Nature", "task_nature"),
    ("Task Budget", "task_budget"),
)

physics_desc = {}
physics_desc["DTI"] = """Diffusion Tensor Imaging (Cholesky parameterization):
- The network outputs 7 parameter maps: S0 (baseline signal intensity) + 6 Cholesky factor components (L11, L21, L22, L31, L32, L33).
- The diffusion tensor D is reconstructed as D = L L^T, where L is a lower-triangular matrix with strictly positive diagonal elements (L11, L22, L33 > 0). This guarantees D is symmetric positive definite by construction.
- Forward model: S = S0 * exp(-b * g^T * D * g), where g is the known diffusion encoding direction and b is the known diffusion weighting."""

physics_desc["DKI"] = """Diffusion Kurtosis Imaging (Cholesky parameterization):
- The network outputs 22 parameter maps: S0 (baseline signal intensity) + 6 Cholesky factor components (L11, L21, L22, L31, L32, L33) + 15 kurtosis components (W1111, W2222, W3333, W1112, W1113, W1222, W1333, W2223, W2333, W1122, W1133, W2233, W1123, W1223, W1233).
- The diffusion tensor D is reconstructed from the Cholesky factors as D = L L^T, which guarantees the diffusion tensor is symmetric positive definite by construction.
- The kurtosis terms model the non-Gaussian deviation of diffusion and are combined with the diffusion tensor terms in the forward model to predict the signal under DKI.
- The constrain term is used to keep the kurtosis tensor physically valid, encouraging the corresponding fourth-order form to be positive for almost all directions and reducing unphysical kurtosis configurations.
- Forward model: the predicted signal is generated from the DTI second-order term together with the DKI fourth-order kurtosis correction, using the known diffusion encoding direction g and weighting b."""

physics_desc["NODDI"] = """Neurite Orientation Dispersion and Density Imaging (Watson-distributed PGSE model):
- The network outputs 5 parameter maps: fiso (isotropic/free-water volume fraction), ficvf (intra-cellular volume fraction inside the non-isotropic tissue pool), kappa (Watson concentration), theta, and phi (fiber-direction spherical angles in radians).
- The NODDI physics class declares require_S0=True. The framework estimates S0 from the measured b0 channels and passes it to output_to_param, which prepends S0 before the forward model.
- The forward model uses a three-compartment NODDI signal: isotropic/free-water diffusion, extra-cellular hindered diffusion, and intra-cellular restricted stick diffusion. The anisotropic tissue signal is mixed by ficvf, then the tissue and isotropic signals are mixed by fiso.
- Orientation dispersion is modeled by a Watson distribution. The hindered compartment uses Watson-averaged parallel and perpendicular diffusivities, and the restricted compartment uses the Watson spherical-harmonic stick expression with Legendre terms and erfi/Dawson-based coefficients.
- Forward model: S = S0 * [(1 - fiso) * ((1 - ficvf) * E_hindered + ficvf * E_restricted) + fiso * E_iso], using the known diffusion gradient direction g and weighting b."""

HardRequirements = Union[dict, str, list]


def default_hard_requirements() -> HardRequirements:
    return "do not set data.dataset to patchwise"
    # return ""


def build_task_hard_requirements(task_info: dict) -> dict:
    return {
        "physics.name": task_info["model"],
    }


# ---------------------------------------------------------------------------
# Prompt builders called by agent/prompts/architect.py
# ---------------------------------------------------------------------------

def build_task_description(cls, dataset_features, model) -> str:
    desc = f"Dataset Features:\n{json.dumps(dataset_features, indent=2, ensure_ascii=False)}\n\n"
    desc += f"Physical Model: {physics_desc[model]}\n\n"
    return desc
