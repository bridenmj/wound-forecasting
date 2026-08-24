# Repository extraction plan

## Phase 1: frozen provenance

- Preserve the three final notebooks privately in Google Drive without editing
  them; do not distribute them through Git.
- Preserve captured source trees and source-location manifests.
- Record canonical checkpoint identities and hashes.
- Export final metrics to portable JSON/CSV.
- Add only final selected manuscript and supplemental figures.

## Phase 2: corrected reusable evaluation

- Extract the final KID implementation. (Complete.)
- Extract targetwise PSNR/SSIM from the final LLaMA notebook. (Complete.)
- Normalize horizon keys to `H1` through `H4` at the public boundary.
- Add pool-shape, target-alignment, and aggregation tests.
- Parameterize data roots and remove Colab-specific absolute paths.

## Phase 3: model packages

- Extract DyneODE's variable-context GRU encoder and conditioned ODE solver.
- Isolate LLaMA-Adapter dataset, prompting, generation, and evaluation code.
- Package River with its `Shared.vq` dependency or replace the absolute cross-project import.
- Define environment files from actual imports and verified versions.

## Phase 4: reproducible commands

- Add dataset-manifest preparation commands.
- Add one evaluation entry point per model.
- Add figure-generation scripts that consume saved results rather than rerunning inference.
- Verify a clean checkout without raw data or local Google Drive paths.
