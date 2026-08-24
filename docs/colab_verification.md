# Colab verification protocol

Run verification in layers. A successful package import does not prove that a
historical checkpoint or dataset layout is compatible.

## 1. Fresh runtime and package tests

Use a GPU runtime, clone the public repository, and install it from the checkout:

```python
!nvidia-smi
!git clone YOUR_REPOSITORY_URL wound-forecasting
%cd wound-forecasting
!python -m pip install -U pip
!python -m pip install -e ".[dev,viz]"
!python scripts/verify_environment.py --device cuda
!pytest -q
```

Expected outcome: `Environment verification: PASS` followed by passing tests.
This layer uses synthetic tensors and requires neither data nor checkpoints.

## 2. Artifact identity checks

Mount Drive or authenticate to a private artifact host. Before loading a weight,
compare its size and SHA-256 value with `artifacts/checkpoints/manifest.tsv`:

```python
from google.colab import drive
drive.mount("/content/drive")

!sha256sum /path/to/checkpoint.pth
```

Do not proceed when the architecture/configuration and checkpoint identity do
not match. In particular, River's final regressor input dimension must match the
final YAML rather than an older sweep configuration.

## 3. Model construction and checkpoint loading

Test each model independently:

1. Construct the final architecture from its recorded configuration.
2. Load the checkpoint strictly.
3. Print missing/unexpected keys and treat either as a failure unless explicitly
   documented.
4. Run one deterministic inference example.
5. Confirm output shape, finite values, and image range.

Do not start a complete held-out evaluation until this succeeds for one sample.

## 4. Real-data smoke test

For each model, use one authorized eight-frame trajectory and verify:

- the pig and wound identifiers are the intended split;
- context and target indices are printed explicitly;
- generated and target tensors have identical image shapes;
- decoded images fall in `[0,1]`;
- the selected random seed is printed and saved;
- a ground-truth/reconstruction/prediction grid is visually plausible.

This is a compatibility test, not a paper metric.

## 5. Evaluation regression

Save one prediction per unique held-out target as:

```python
torch.save(
    {"fake_pools": fake_pools, "real_pools": real_pools},
    "model_pools.pt",
)
```

Then use the same public evaluation command for every model:

```python
!python scripts/evaluate_pools.py model_pools.pt \
    --output results/metrics/model.json \
    --device cuda
```

Confirm combined target counts of `60, 40, 20, 10` for H1-H4 and `130` overall.
Compare the resulting JSON against the finalized paper numbers before calling
the model reproducible.

## 6. Minimal training test

Only after inference passes, run a tiny training job with:

- one or two trajectories;
- one optimizer step;
- checkpoint save and reload;
- fixed seed;
- experiment tracking disabled initially.

The objective is to catch imports, paths, device placement, and serialization
errors. It is not intended to reproduce convergence.

## Hugging Face artifact policy

Use separate private repositories for weights and any authorized data. Prefer a
small synthetic or explicitly shareable example dataset for public testing.

Before uploading wound images, confirm in writing that the study consent, IRB
or ethics determination, data-use agreement, and institutional policy permit
redistribution. De-identification alone does not automatically authorize public
release. If redistribution is not explicitly permitted, keep the data in the
approved storage environment and publish only manifests, schemas, synthetic
examples, and access instructions.

For model weights, verify that licenses for the LLaMA base model, adapters,
StyleGAN/e4e components, River, VQ-MUSE, and all bundled dependencies permit the
specific form of redistribution. Upload only project-owned deltas or adapters
when the base-model license requires users to obtain upstream weights directly.

Record every hosted artifact's repository, revision/commit, filename, byte size,
and SHA-256 value in `artifacts/checkpoints/manifest.tsv`. Temporary hosting for
several months is useful for verification, but the paper should not depend on an
undocumented expiring URL.

