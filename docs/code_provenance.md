# Code provenance and authorship boundary

This repository separates project-specific wound-forecasting work from
externally managed upstream research implementations.

## Project-specific public modules

- `src/wound_forecasting/dyneode*.py`: final variable-context DyneODE model,
  latent dataset, optimization, checkpoint, and inference workflows.
- `src/wound_forecasting/llama_*.py`: aligned-burst wound tokenization,
  autoregressive training dataset, adapter-only serialization, deterministic
  inference, and evaluation built around the upstream LVM API.
- `src/wound_forecasting/river.py`: strict project-weight loading and aligned
  variable-length inference built around the external River implementation.
- `src/wound_forecasting/metrics.py`: corrected common paper evaluation.
- `src/wound_forecasting/manifests.py`: explicit subject-level split handling.

## External implementations

The upstream LLaMA-Adapter/LVM, River, StyleGAN, e4e, and VQ-MUSE source trees
are not copied into this repository. Their identities and checkpoint
dependencies are recorded in the documentation and artifact manifest. Public
project modules expose only the wound-forecasting additions and shared
evaluation boundary; upstream work should not be interpreted as original
project code. Runtime boundary modules require users to supply those source
trees and externally licensed weights explicitly.

## Private materials

Raw wound images, trained weights, latent inversions, experiment-tracking data,
exploratory notebooks, and historical source archives are not distributed
through Git.
