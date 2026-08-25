# Repository extraction plan

## Phase 1: private provenance and public identities

- Preserve the three final notebooks privately in Google Drive without editing
  them; do not distribute them through Git.
- Preserve historical source trees privately and record upstream identities in
  the public provenance documentation.
- Record canonical checkpoint identities and hashes.
- Export final metrics to portable JSON/CSV.
- Add only final selected manuscript and supplemental figures.

## Phase 2: corrected evaluation — complete

- Extract the final KID implementation. (Complete.)
- Extract targetwise PSNR/SSIM from the final LLaMA notebook. (Complete.)
- Normalize horizon keys to `H1` through `H4` at the public boundary. (Complete.)
- Add pool-shape, target-alignment, and aggregation tests. (Complete.)
- Parameterize data roots and remove Colab-specific absolute paths. (Complete.)

## Phase 3: model packages — complete

- Extract DyneODE's variable-context GRU encoder, training, checkpoint, and
  conditioned ODE workflows. (Complete.)
- Isolate LLaMA-Adapter dataset, prompting, adapter training, generation, and
  evaluation code. (Complete.)
- Keep River and `Shared.vq` external while accepting their locations through
  explicit command-line arguments. (Complete.)
- Define dependencies from actual imports. (Complete.)

## Phase 4: reproducible commands — implemented; real-artifact regression pending

- Add dataset-manifest preparation commands.
- Add one training and evaluation entry point per model. (Complete.)
- Add figure-generation scripts that consume saved results rather than
  rerunning inference.
- Verify a clean checkout without raw data or local Google Drive paths.
  (Synthetic tests complete; one-trajectory Colab regressions remain.)
