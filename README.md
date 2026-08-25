# Wound Forecasting

Research code and paper-era provenance for longitudinal wound-image forecasting with LLaMA-Adapter, DyneODE, and River.

## Repository status

This repository is organized in two layers:

1. `paper_snapshot/` preserves supporting source used to produce the manuscript
   results; the complete exploratory notebooks are retained privately in Google
   Drive.
2. `src/` contains the cleaned, reusable implementation extracted from that snapshot.

The paper snapshot is intentionally frozen before any notebook cleanup or path refactoring. It is not expected to run on a new machine without the externally managed datasets and model weights described under `artifacts/`.

## Current structure

```text
wound-forecasting/
├── paper_snapshot/
│   ├── notebooks/       # Optional local copies; ignored by Git
│   ├── evaluation/      # Paper-era standalone evaluation helpers
│   └── source/          # Source trees captured from Google Drive
├── artifacts/
│   └── checkpoints/     # Metadata only; weights are not committed
├── docs/                # Audit and extraction documentation
├── results/
│   ├── figures/         # Final selected publication figures
│   └── metrics/         # Portable final results
├── scripts/             # Reproducible command-line entry points
├── src/                 # Clean shared Python package
└── tests/               # Lightweight correctness checks
```

## Important provenance warning

The standalone `paper_snapshot/evaluation/metrics.py` predates the final
targetwise PSNR/SSIM correction. Use `wound_forecasting.metrics` for new work.
Frozen files under `paper_snapshot/` are provenance records and are not edited
in place.

## Public-release scope

The repository is intended to show the full scientific design without requiring
readers to reverse-engineer exploratory notebooks:

- model definitions, final configurations, split logic, and evaluation code are
  public;
- compact reproducible entry points will live under `scripts/`;
- the complete paper-era notebooks remain privately archived as provenance;
- raw clinical images, checkpoints, credentials, and machine-specific paths are
  excluded.

The cleaned API currently includes subject-level manifest selection, KID, and
targetwise PSNR/SSIM, the final variable-context DyneODE architecture, and the
project-specific LLaMA-Adapter generation boundary. River's captured training
implementation is exposed through a parameterized public entry point.

## Install for development

```bash
python -m pip install -e ".[dev]"
pytest
```

For a clean Google Colab verification—from synthetic import tests through one
real trajectory, checkpoint loading, metric regression, and a one-step training
smoke test—follow [docs/colab_verification.md](docs/colab_verification.md).

## Public entry points

Evaluate saved, one-prediction-per-target image pools consistently across all
three models:

```bash
python scripts/evaluate_pools.py pools.pt --output results/metrics/model.json
```

The input file must contain `fake_pools` and `real_pools` dictionaries keyed by
forecast horizon. Each corresponding tensor must be aligned and shaped
`[N, 3, H, W]`.

Launch the final River training configuration without a hard-coded Colab path:

```bash
python scripts/train_river.py \
  --run-name final-river \
  --config configs/river_final.yaml \
  --data-root /path/to/prepared/data \
  --num-gpus 1
```

Create an edge-to-edge qualitative grid from a saved eight-frame tensor:

```bash
python -m pip install -e ".[viz]"
python scripts/plot_sequence_grid.py sequence.pt --output sequence.png
```

## Model-specific public code

- `wound_forecasting.dyneode` contains the time-aware GRU context encoder,
  context-conditioned vector field, broadcast-W validation, and conditioned ODE
  integration from the final variable-context experiment.
- `wound_forecasting.llama_adapter` contains only this project's deterministic
  suffix-generation, strict adapter-delta loading, and VQ decoding additions.
  The upstream LLaMA-Adapter project is not presented as original work.
- `paper_snapshot/source/river` preserves the River implementation and final
  project changes; `configs/river_final.yaml` and `scripts/train_river.py`
  provide the cleaner public interface.

## Data and weights

Raw wound images, latent inversions, HDF5 shards, W&B runs, and model checkpoints
are deliberately excluded from Git. See `artifacts/checkpoints/manifest.tsv`
and `docs/source_snapshot_audit.md`.

The wound-specific LLaMA artifact is distributed as an adapter delta rather
than a combined 7B checkpoint. Construct the matching architecture from the
upstream `Emma02/LVM_ckpts` base, then overlay
`wound_llama_adapter_delta_v1.pth` with
`wound_forecasting.llama_adapter.load_adapter_delta`. The extracted artifact
contains 84 project-trained tensors. In a fixed-seed regression against the
private archival checkpoint, base plus adapter reproduced the generated token
sequence exactly (zero differing token positions). The full base checkpoint,
private verification inputs, and reference tokens are not committed.

VQ-MUSE is also an external dependency and should be obtained from
`Emma02/vqvae_ckpts`; this repository does not mirror its weights.
