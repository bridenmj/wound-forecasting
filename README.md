# Longitudinal Wound Progression Modeling Under Sparse Sampling: A Comparative Study of Discrete and Continuous Generative Frameworks

Unified codebase for longitudinal wound trajectory forecasting under sparse and irregular clinical sampling.  
This repository compares three complementary generative paradigms under a consistent held-out evaluation protocol:

- **River-CFM**: sparsely conditioned flow matching for video prediction (adapted to wound sequences)
- **DyneODE**: latent Neural ODE extrapolation in a generative latent space
- **LLaMA-Adapter (LVM)**: vision-language model–based discrete token forecasting (image-token sequence modeling)

This repo emphasizes **reproducible, apples-to-apples evaluation** across methods using shared metrics (**PSNR / SSIM / KID**) and consistent horizon-based testing.

---

## Repository Layout

- `external/`  
  Upstream repositories (vendored) with minimal patches for this project:
  - `external/river/`
  - `external/DyneODE/`
  - `external/LLaMA-Adapter/`

- `src/`  
  Project “glue” code (datasets, evaluation wrappers, shared utilities):
  - `src/shared/` (VQ-Muse wrapper, manifest utilities, etc.)
  - `src/metrics/` (shared metrics: KID / PSNR / SSIM)
  - `src/river_cfm/`, `src/dyneode/`, `src/llama_adapter/` (method-specific wrappers)

- `configs/`  
  YAML configs for each method.

- `scripts/`  
  Thin CLI entrypoints (training/eval wrappers).

- `data/` *(not included in git)*  
  Instructions for where to place images / latents / HDF5.

- `checkpoints/` *(not included in git)*  
  Model checkpoints and pretrained weights (paths provided by the user).

---

## Status

The initial commit provides scaffolding plus core evaluation/metric utilities.  
Over the next few days, this will be populated with:
- standalone `train.py` / `eval.py` wrappers for each method
- a single “evaluate-all” script/notebook that reproduces the paper tables/figures
- clearer configuration defaults and documented data paths

---

## Data Setup

See `data/README.md` for the expected directory structure.

At a high level, the project uses:
- wound image sequences (RGB)
- DyneODE latent `.pt` files (aligned with images)
- HDF5 video datasets for River-CFM

---

## Evaluation Metrics

We report:
- **PSNR** ↑ and **SSIM** ↑ for image similarity
- **KID** ↓ for distributional realism

Shared implementations live in `src/metrics/metrics.py`.

---

## Running (draft)

Scripts will be finalized shortly.