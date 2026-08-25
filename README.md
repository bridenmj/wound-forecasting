# Wound Forecasting

Implementation and evaluation framework for longitudinal wound-image forecasting
with LLaMA-Adapter, DyneODE, and River.

## Repository status

This repository provides model components, configuration, dataset split logic,
evaluation metrics, visualization utilities, and command-line workflows for
training and evaluating longitudinal wound-forecasting systems.

## Current structure

```text
wound-forecasting/
├── artifacts/
│   └── checkpoints/     # Metadata only; weights are not committed
├── docs/                # Architecture and reproducibility documentation
├── results/
│   ├── figures/         # Final selected publication figures
│   └── metrics/         # Portable final results
├── scripts/             # Reproducible command-line entry points
├── src/                 # Python package
└── tests/               # Lightweight correctness checks
```

## Project scope

The repository exposes the components required to understand, test, and extend
the forecasting system:

- model definitions, final configurations, split logic, and evaluation code;
- reproducible command-line entry points under `scripts/`;
- automated tests for core model, manifest, and metric behavior;
- external artifact boundaries for datasets and model weights.

Credentials, machine-specific paths, datasets, and model checkpoints are not
committed to the source repository.

The package includes subject-level manifest selection; the final variable-
context DyneODE architecture, training loop, and inference pipeline; the
wound-specific LLaMA-Adapter token dataset, adapter training, inference, and
evaluation pipeline; River checkpoint loading and evaluation; and shared KID
and targetwise PSNR/SSIM implementations. Upstream LVM, VQ-MUSE, StyleGAN, and
River implementations remain explicit external dependencies.

## Install for development

```bash
python -m pip install -e ".[dev]"
pytest
```

For Google Colab verification—from synthetic import tests through one
real trajectory, checkpoint loading, metric regression, and a one-step training
smoke test—follow [docs/colab_verification.md](docs/colab_verification.md).

## Public entry points

Train and evaluate the final variable-context DyneODE model:

```bash
python scripts/train_dyneode.py --help
python scripts/evaluate_dyneode.py --help
```

Train the wound-specific LLaMA adapter without saving a duplicate 7B base
checkpoint, then evaluate its released adapter delta:

```bash
python scripts/train_llama_adapter.py --help
python scripts/evaluate_llama_adapter.py --help
```

River uses its separately obtained implementation and VQ-MUSE dependency:

```bash
python scripts/train_river.py --help
python scripts/evaluate_river.py --help
```

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
  integration from the final variable-context experiment. The adjacent data,
  training, inference, and StyleGAN-boundary modules implement the complete
  project workflow.
- `wound_forecasting.llama_data`, `llama_training`, `llama_inference`, and
  `llama_adapter` implement this project's aligned-burst tokenization,
  autoregressive dataset, adapter-only checkpoints, deterministic generation,
  and corrected one-prediction-per-target evaluation. The upstream
  LLaMA-Adapter project is not presented as original work.
- `wound_forecasting.river` provides strict River-only checkpoint loading and
  aligned variable-length evaluation. `configs/river_final.yaml` and the two
  River scripts provide path-parameterized interfaces to the external River
  implementation.

## Evaluation contract

All paper-comparison evaluators use the same unit of analysis: one generated
image for each unique target image. Relative forecast positions are reported as
H1 through H4 even where LLaMA internally addresses absolute sequence positions
4 through 7. The expected combined held-out counts are 60, 40, 20, and 10,
respectively (130 targets overall). PSNR and SSIM are averaged over those unique
prediction-target pairs; pooled KID is weighted by the available target count.

## Data and weights

Raw wound images, latent inversions, HDF5 shards, W&B runs, and model checkpoints
are deliberately excluded from Git. Model artifacts and prepared data are
hosted separately on Hugging Face so that the source repository remains small.
See `artifacts/checkpoints/manifest.tsv` for the canonical artifact identities
and external dependencies.

| Artifact | Hugging Face repository | Contents |
|---|---|---|
| LLaMA-Adapter | [bridenmj/wound-llama-adapter](https://huggingface.co/bridenmj/wound-llama-adapter) | Wound-specific 84-tensor adapter delta |
| StyleGAN | [bridenmj/wound-stylegan](https://huggingface.co/bridenmj/wound-stylegan) | Wound-domain StyleGAN generator and configuration |
| DyneODE | [bridenmj/wound-dyneode](https://huggingface.co/bridenmj/wound-dyneode) | Final variable-context DyneODE checkpoint and configuration |
| River | [bridenmj/wound-river](https://huggingface.co/bridenmj/wound-river) | River-only weights and final configuration; VQ-MUSE weights are excluded |
| Processed data | [bridenmj/porcine-wound-forecasting-processed](https://huggingface.co/datasets/bridenmj/porcine-wound-forecasting-processed) | Processed images, DyneODE inversion latents, and River HDF5 data |

These repositories remain private while release permissions and licensing are
being finalized. The links will become accessible when their respective
repositories are published.

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
[`Emma02/vqvae_ckpts`](https://huggingface.co/Emma02/vqvae_ckpts); this
repository does not mirror its weights. The matching LVM base checkpoint is
available from [`Emma02/LVM_ckpts`](https://huggingface.co/Emma02/LVM_ckpts).
