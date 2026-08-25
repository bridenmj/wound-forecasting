# Code provenance and authorship boundary

This repository separates project-specific wound-forecasting work from
externally managed upstream research implementations.

## Project-specific public modules

- `src/wound_forecasting/dyneode.py`: final variable-context DyneODE extension.
- `src/wound_forecasting/llama_adapter.py`: wound-sequence generation boundary
  built around the upstream adapter API.
- `src/wound_forecasting/metrics.py`: corrected common paper evaluation.
- `src/wound_forecasting/manifests.py`: explicit subject-level split handling.

## External implementations

The upstream LLaMA-Adapter/LVM, River, StyleGAN, e4e, and VQ-MUSE source trees
are not copied into this repository. Their identities and checkpoint
dependencies are recorded in the documentation and artifact manifest. Public
project modules expose only the wound-forecasting additions and shared
evaluation boundary; upstream work should not be interpreted as original
project code.

## Private materials

Raw wound images, trained weights, latent inversions, experiment-tracking data,
exploratory notebooks, and historical source archives are not distributed
through Git.
