# Code provenance and authorship boundary

This repository separates project-specific wound-forecasting work from
externally managed upstream research implementations.

## Project-specific public modules

- `src/wound_forecasting/dyneode*.py`: final variable-context DyneODE model,
  latent dataset, optimization, checkpoint, and inference workflows.
- `src/wound_forecasting/llama_*.py`: aligned-burst wound tokenization,
  autoregressive training dataset, adapter-only serialization, deterministic
  inference, and evaluation built around the upstream LVM API.
- `src/wound_forecasting/river*.py`: final self-conditioned transformer vector
  field, conditional-flow-matching model, HDF5 sequence data, optimization,
  adapter-only checkpointing, and aligned variable-length inference.
- `src/wound_forecasting/metrics.py`: corrected common paper evaluation.
- `src/wound_forecasting/manifests.py`: explicit subject-level split handling.

## External implementations

Installing this repository installs all modules under
`src/wound_forecasting/`, including the project-specific DyneODE architecture,
data handling, training and inference code; the wound-specific LLaMA-Adapter
data, training, serialization and inference code; River architecture, HDF5
data, training, checkpointing, and evaluation; and the shared manifest and
metric implementations.

The optional dependency groups in `pyproject.toml` install ordinary Python
packages needed by those workflows. They do **not** clone the upstream
LLaMA-Adapter/LVM, River, StyleGAN2, e4e, or VQ-MUSE research repositories, and
they do not download externally licensed base weights.

End-to-end execution therefore requires the applicable external runtime:

| Workflow | Included here | Supplied separately at runtime |
|---|---|---|
| LLaMA-Adapter | Wound token dataset, adapter training, adapter-delta loading, deterministic suffix generation, and evaluation | LVM/LLaMA-Adapter source, LVM base weights, and VQ-MUSE source/weights |
| DyneODE | Context encoder, conditioned Neural ODE, latent data handling, training, inference, and evaluation | StyleGAN2 source and the released wound-domain generator |
| River | Wound-specific self-conditioned vector field, CFM model, HDF5 data handling, training, checkpointing, and evaluation | VQ-MUSE source/weights |

The released DyneODE inversion latents can be consumed directly, so e4e is not
required to reproduce training or evaluation from those representations. e4e
is needed only to create new inversions from additional images.

Runtime boundary modules such as `lvm_upstream.py` and `stylegan.py` make these
dependencies explicit by accepting source and checkpoint paths. Upstream work
must not be interpreted as original project code and remains governed by its
own license and terms.

## Artifacts outside Git

Large trained weights, processed images, latent inversions, and River HDF5 data
are hosted in the project Hugging Face repositories rather than committed to
Git. Exploratory notebooks, experiment-tracking data, and historical source
archives are not part of the public source release. Canonical artifact names
and locations are recorded in `artifacts/checkpoints/manifest.tsv`.
