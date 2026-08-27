# Longitudinal Wound Progression Modeling Under Sparse Sampling

### A Comparative Study of Discrete and Continuous Generative Frameworks

[![Models](https://img.shields.io/badge/%F0%9F%A4%97%20Models-Wound%20Forecasting-yellow)](https://huggingface.co/bridenmj)
[![Dataset](https://img.shields.io/badge/%F0%9F%A4%97%20Dataset-Processed%20Porcine%20Wounds-yellow)](https://huggingface.co/datasets/bridenmj/porcine-wound-forecasting-processed)
[![Python](https://img.shields.io/badge/Python-%E2%89%A53.10-3776AB.svg?logo=python&logoColor=white)](pyproject.toml)
[![License](https://img.shields.io/badge/License-CC%20BY--NC%204.0-green.svg)](LICENSE.txt)

Forecasting wound healing from longitudinal images is challenging because
follow-up observations are sparse and irregular, available datasets are small,
and tissue repair is governed by complex biological processes. This project
formulates wound progression as an image-sequence prediction problem and asks
how different generative modeling strategies balance reconstruction fidelity
against perceptual realism.

We compare three efficient generative frameworks—**LLaMA-Adapter**,
**Sparsely Conditional Flow Matching (River CFM)**, and a
**StyleGAN2-based DyneODE**—for sequential wound-image forecasting. Models are
trained on longitudinal porcine wound images collected over a 21-day healing
period and conditioned on early observations to generate future trajectories.
Performance is evaluated across forecast horizons using Kernel Inception
Distance (KID), Peak Signal-to-Noise Ratio (PSNR), Structural Similarity Index
Measure (SSIM), and qualitative trajectory analysis.

Together, the experiments characterize a central tradeoff in small-data wound
forecasting: models with stronger pixelwise reconstruction scores may produce
oversmoothed or mean-like futures, while autoregressive image-token modeling
can better preserve localized texture and perceptually plausible progression.
The repository provides the model implementations, training and evaluation
entry points, final configurations, subject-level split logic, processed data,
and independently hosted model weights used in the study.

## Highlights

- Three generative forecasting approaches spanning autoregressive image-token
  modeling, continuous latent dynamics, and conditional flow matching.
- Subject-level evaluation on held-out pigs, with explicit prevention of
  train/test leakage.
- Support for variable-length trajectories, irregular observation times, and
  variable-context extrapolation.
- A shared evaluation contract for KID, targetwise PSNR, and targetwise SSIM.
- Reproducible releases of project-trained weights and processed model inputs.
- Automated correctness checks plus real-data integration workflows for every
  released model stack.

## Forecasting frameworks

| Framework | Representation | Forecasting mechanism | Released artifact |
|---|---|---|---|
| **LLaMA-Adapter** | VQ-MUSE image tokens | Autoregressive multimodal transformer with a wound-specific adapter | [Adapter delta](https://huggingface.co/bridenmj/wound-llama-adapter) |
| **DyneODE** | StyleGAN latent space | Time-aware GRU context encoder and context-conditioned Neural ODE | [DyneODE checkpoint](https://huggingface.co/bridenmj/wound-dyneode) |
| **River** | VQ-MUSE latent representation | Conditional flow matching for future-frame generation | [River weights](https://huggingface.co/bridenmj/wound-river) |

The wound-domain StyleGAN generator used by DyneODE is released separately at
[`bridenmj/wound-stylegan`](https://huggingface.co/bridenmj/wound-stylegan).

## Held-out evaluation

All three frameworks use the same unit of analysis: one generated image for
each unique target image. Relative forecast positions are reported as H1–H4.
The combined held-out evaluation contains 60, 40, 20, and 10 targets at those
horizons, respectively, for 130 prediction–target pairs overall.

| Model | Overall KID ↓ | PSNR ↑ | SSIM ↑ |
|---|---:|---:|---:|
| LLaMA-Adapter | **0.0605** | 17.0662 | 0.3211 |
| DyneODE | 0.1880 | **18.9879** | **0.3755** |
| River CFM | 0.1592 | 17.2617 | 0.2733 |

KID is recomputed from the pooled real and generated image sets. PSNR and SSIM
are averaged over aligned, unique prediction–target pairs rather than over
overlapping sequence windows.

## Released artifacts

| Artifact | Repository | Contents |
|---|---|---|
| LLaMA-Adapter | [`bridenmj/wound-llama-adapter`](https://huggingface.co/bridenmj/wound-llama-adapter) | Verified 84-tensor wound-specific adapter delta |
| StyleGAN | [`bridenmj/wound-stylegan`](https://huggingface.co/bridenmj/wound-stylegan) | Wound-domain generator and architecture configuration |
| DyneODE | [`bridenmj/wound-dyneode`](https://huggingface.co/bridenmj/wound-dyneode) | Final variable-context checkpoint and configuration |
| River | [`bridenmj/wound-river`](https://huggingface.co/bridenmj/wound-river) | River-only trained weights and final configuration |
| Processed data | [`bridenmj/porcine-wound-forecasting-processed`](https://huggingface.co/datasets/bridenmj/porcine-wound-forecasting-processed) | Processed images, DyneODE inversions, River HDF5 data, and LLaMA prompts |

Large model artifacts and datasets are intentionally hosted outside Git. Their
canonical filenames and dependency boundaries are recorded in
[`artifacts/checkpoints/manifest.tsv`](artifacts/checkpoints/manifest.tsv).

## Installation

This project is an integration layer over the complete upstream research
frameworks. Install the upstream source trees first, then install this
repository on top. The upstream projects remain governed by their own licenses.

### 1. Clone the project and upstream frameworks

```bash
git clone https://github.com/bridenmj/wound-forecasting.git
mkdir -p wound-forecasting/upstream

# LLaMA-Adapter primitives
git clone https://github.com/OpenGVLab/LLaMA-Adapter.git \
  wound-forecasting/upstream/LLaMA-Adapter

# LVM VQ-MUSE implementation used by LLaMA-Adapter and River
git clone https://huggingface.co/spaces/Emma02/LVM \
  wound-forecasting/upstream/LVM

# Complete upstream River implementation
git clone https://github.com/Araachie/river.git \
  wound-forecasting/upstream/river

# Complete upstream DyneODE/StyleGAN2 implementation
git clone https://github.com/weihaox/dynode.git \
  wound-forecasting/upstream/dynode
```

The upstream checkouts are retained intact. Project-trained modules and
behavioral changes are implemented under `src/wound_forecasting/`; users do not
need to copy project files into the upstream repositories.

### 2. Install the wound-forecasting package

```bash
cd wound-forecasting

# Choose one workflow.
python -m pip install -e ".[dyneode]"
python -m pip install -e ".[river]"
python -m pip install -e ".[llama]"

# Or install the dependencies for every workflow.
python -m pip install -e ".[all]"
```

For repository development and tests, add the development dependencies:

```bash
python -m pip install -e ".[all,dev]"
python -m ruff check src scripts tests
python -m pytest -q
```

### 3. Download model weights and processed data

Project-trained artifacts are downloaded from the repositories listed under
[Released artifacts](#released-artifacts). The compatible external LVM and
VQ-MUSE checkpoints are obtained separately from
[`Emma02/LVM_ckpts`](https://huggingface.co/Emma02/LVM_ckpts) and
[`Emma02/vqvae_ckpts`](https://huggingface.co/Emma02/vqvae_ckpts).

The command-line tools accept explicit source and checkpoint paths. With the
layout above, the important source arguments are:

| Workflow | Argument | Path |
|---|---|---|
| LLaMA-Adapter | `--lvm-source` | `upstream/LLaMA-Adapter` |
| LLaMA-Adapter | `--vq-source` | `upstream/LVM` |
| River | `--vq-source` | `upstream/LVM` |
| DyneODE | `--stylegan-source` | `upstream/dynode/code` |

The pretrained checkpoints are not copied into the source trees. Pass their
downloaded directories or files through `--llama-checkpoint-dir`,
`--vq-checkpoint-dir`, and the applicable checkpoint arguments.

## Training and evaluation

Each final framework has a path-parameterized command-line entry point:

```bash
# DyneODE
python scripts/train_dyneode.py --help
python scripts/evaluate_dyneode.py --help

# LLaMA-Adapter
python scripts/train_llama_adapter.py --help
python scripts/evaluate_llama_adapter.py --help

# River
python scripts/train_river.py --help
python scripts/evaluate_river.py --help
```

The corresponding paper configurations are stored in [`configs/`](configs/).
The entry points combine the installed upstream frameworks with the
wound-specific implementations in `src/wound_forecasting/` and the released
project checkpoints. River and LLaMA-Adapter share the compatible VQ-MUSE
runtime; DyneODE uses the StyleGAN2 implementation from the complete upstream
DyneODE checkout, the released wound-domain generator, and the released
inversion latents. These boundaries are documented in
[`docs/code_provenance.md`](docs/code_provenance.md).

### Shared metric evaluation

Previously generated image pools can be evaluated consistently across models:

```bash
python scripts/evaluate_pools.py pools.pt \
  --output results/metrics/model.json
```

The input must contain aligned `fake_pools` and `real_pools` dictionaries keyed
by forecast horizon, with tensors shaped `[N, 3, H, W]`.

### Sequence visualization

Create an edge-to-edge eight-frame sequence grid without notebook dependencies:

```bash
python -m pip install -e ".[viz]"
python scripts/plot_sequence_grid.py sequence.pt --output sequence.png
```

## Repository structure

```text
wound-forecasting/
├── artifacts/checkpoints/   # Artifact identities and external locations
├── configs/                 # Final model configurations
├── docs/                    # Provenance and reproducibility documentation
├── results/                 # Portable figures and metric outputs
├── scripts/                 # Training, evaluation, and plotting entry points
├── src/wound_forecasting/   # Reusable project implementation
└── tests/                   # Core correctness and regression tests
```

## Reproducibility

The automated suite validates model construction, adapter-delta loading,
subject-level manifest behavior, target alignment, horizon aggregation, and
metric calculations. The full Colab verification procedure additionally tests
strict checkpoint loading and one real held-out trajectory through each stack:

1. DyneODE + wound-domain StyleGAN;
2. River + VQ-MUSE;
3. LVM base + wound-specific LLaMA adapter + VQ-MUSE.

See [`docs/colab_verification.md`](docs/colab_verification.md) for the complete
verification sequence.

## Data provenance

The processed release is derived from the public longitudinal porcine
wound-healing dataset described by Isseroff and colleagues:

- [Dataset publication](https://www.nature.com/articles/s41597-025-05921-w)
- [Dryad source](https://doi.org/10.5061/dryad.0rxwdbsbr)
- [CC0 1.0 Universal source dedication](https://creativecommons.org/publicdomain/zero/1.0/)

The source dedication, project code license, model-weight licenses, and
third-party dependency terms are distinct. Consult each release card and
upstream repository before reuse.

## Intended use and limitations

This repository is intended for research on longitudinal image forecasting,
generative modeling, and wound-healing progression. The data depict porcine
wounds rather than human clinical cases. Generated images are not clinical
measurements, and the released models are not medical devices and must not be
used for diagnosis, treatment selection, or patient care. 

## License

The original code and project-trained model weights produced by this project
are licensed under the
[Creative Commons Attribution–NonCommercial 4.0 International license](LICENSE.txt)
(CC BY-NC 4.0).

Third-party implementations, base models, tokenizers, and other external
dependencies are not relicensed by this repository and remain subject to their
respective licenses and terms. See
[`docs/code_provenance.md`](docs/code_provenance.md) for project boundaries and
upstream provenance.
