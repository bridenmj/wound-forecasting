# Code provenance and authorship boundary

This repository separates project-specific wound-forecasting work from captured
upstream research implementations.

## Project-specific public modules

- `src/wound_forecasting/dyneode.py`: final variable-context DyneODE extension.
- `src/wound_forecasting/llama_adapter.py`: wound-sequence generation boundary
  built around the upstream adapter API.
- `src/wound_forecasting/metrics.py`: corrected common paper evaluation.
- `src/wound_forecasting/manifests.py`: explicit subject-level split handling.

## Captured upstream-derived trees

The source under `paper_snapshot/source/llama_adapter` and
`paper_snapshot/source/river` is retained to document the exact implementation
context. Its original licenses and READMEs remain alongside it. It should not be
interpreted as wholly original project code.

## Private materials

Raw wound images, trained weights, latent inversions, experiment-tracking data,
and exploratory notebooks are not distributed through Git.
