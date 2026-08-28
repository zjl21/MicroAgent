# MicroAgent

MicroAgent generates a data-specific configuration and performs self-supervised microstructure quantification from diffusion MRI.

The code is released under the [MIT License](LICENSE).

## Overview

<p align="center">
  <img src="assets/figure-framework.png" alt="MicroAgent framework" width="100%">
</p>

The public workflow is:

```text
prepared DWI -> feature extraction -> Agent config generation with skills
             -> self-supervised training -> whole-volume inference
```

## Installation

### 1. Clone the repository

```bash
git clone REPLACE_WITH_REPOSITORY_URL MicroAgent
cd MicroAgent
```

### 2. Check the system prerequisites

You need an NVIDIA GPU and compatible driver, Git, Conda and [MRtrix3](https://www.mrtrix.org/download/). MicroAgent was tested with the PyTorch CUDA 12.1 runtime.

### 3. Choose an environment installation

#### Option 1: Configure from the environment snapshot

`env/agent_environment.yml` contains the complete package snapshot of the tested environment. Use it as a reference and adapt the CUDA Toolkit, PyTorch, compilers, tiny-cuda-nn, and other platform-specific packages to your own system. 

#### Option 2: Use the prebuilt environment

The prebuilt environment is for Linux x86-64 with the PyTorch CUDA 12.1 runtime. Download the complete archive from the external release link:

```bash
wget https://cloud.tsinghua.edu.cn/f/1ec5c1bb109d45e19495/?dl=1 \
  -O MicroAgent-linux-x86_64-cu121.tar.gz
```

Unpack and relocate the environment:

```bash
mkdir -p MicroAgent-env
tar -xzf MicroAgent-linux-x86_64-cu121.tar.gz -C MicroAgent-env
source MicroAgent-env/bin/activate
conda-unpack
```

## Run the demo

### 1. Set the common configuration

Copy the environment template and edit the LLM setting, MRtrix3 installation path, GPU, and data path:

```bash
cp env_config.example.yaml env_config.yaml
```

All Agent roles use the same OpenAI-compatible API block:

```yaml
api:
  provider: openai
  base_url: https://api.openai.com/v1
  model_name: REPLACE_WITH_MODEL_NAME
```

Keep the API key outside the file:

```bash
export API_KEY="YOUR_API_KEY"
```

### 2. Run the DTI demo with the provided CHCP example data

The bundled input is stored in the layout:

```text
data/CHCP/sub-001/diff-1-6-0/
├── sub-001_diff.nii.gz
├── sub-001_diff.bval
├── sub-001_diff.bvec
├── sub-001_diff_mask.nii.gz
├── sub-001_weight_mask.nii.gz
└── sub-001_evaluate_mask_WM.nii.gz
```

Validate the setup and run:

```bash
python run.py --config env_config --preflight-only
python run.py --config env_config
```

The second command automatically extracts `sub-001_features.json`, applies skills, generates a valid training config, performs self-supervised training, and writes DTI maps to `output_dir`.

If a run is interrupted, execute the same command again. Training resumes from the checkpoint in the configured output directory; a completed checkpoint is reused for inference.

### 3. Run the demo with another prepared DWI

Place exactly one set of the following six files in one directory, give them a common prefix such as `sub-001`, and change `data_dir` in `env_config.yaml`:

| Required name | Contents and requirements |
| --- | --- |
| `<prefix>_diff.nii.gz` | Preprocessed 4-D DWI with shape `(X, Y, Z, N)`. |
| `<prefix>_diff.bval` | `N` b-values corresponding to the DWI volumes; at least one b0 volume with `b < 50 s/mm²` is required. |
| `<prefix>_diff.bvec` | `N` gradient vectors, stored as either `3 × N` or `N × 3`. |
| `<prefix>_diff_mask.nii.gz` | 3-D diffusion-space brain mask used by MRtrix3 denoising and SNR estimation. |
| `<prefix>_weight_mask.nii.gz` | 3-D diffusion-space mask defining the voxels used for signal normalization, training/validation, and final parameter-map inference. For the DTI workflow, use a brain-tissue mask that excludes CSF. |
| `<prefix>_evaluate_mask_WM.nii.gz` | 3-D diffusion-space white-matter mask used  to calculate WM versus non-WM tissue signal features for Agent configuration retrieval. |

## Citation
