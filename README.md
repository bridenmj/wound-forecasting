# Wound Forecasting

Research code and paper-era provenance for longitudinal wound-image forecasting
with LLaMA-Adapter, DyneODE, and River.

## Repository status

This public repository contains the cleaned, reusable implementation and
reproducibility metadata extracted from the research project. Complete
exploratory notebooks and historical source archives remain privately retained
in Google Drive; they are not required for the public package tests or shared
evaluation utilities.

## Current structure

```text
wound-forecasting/
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

## Public-release scope

The repository is intended to show the full scientific design without requiring
readers to reverse-engineer exploratory notebooks:

- model definitions, final configurations, split logic, and evaluation code are
  public;
- compact reproducible entry points will live under `scripts/`;
- the complete paper-era notebooks remain privately archived as provenance;
- raw clinical images, checkpoints, credentials, and machine-specific paths are
  excluded.

The public API currently includes subject-level manifest selection, KID, and
targetwise PSNR/SSIM, the final variable-context DyneODE architecture, and the
project-specific LLaMA-Adapter generation boundary. River's final configuration
and a parameterized launcher are included; the launcher expects the external
River source tree to be supplied explicitly.

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
  --river-source /path/to/river/source \
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
- `configs/river_final.yaml` records the final River configuration, while
  `scripts/train_river.py` provides a path-parameterized interface to an
  externally managed River source checkout.

## Data and weights

Raw wound images, latent inversions, HDF5 shards, W&B runs, and model checkpoints
are deliberately excluded from Git. See `artifacts/checkpoints/manifest.tsv`
for the canonical artifact identities and external dependencies.

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
