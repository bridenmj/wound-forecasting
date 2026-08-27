# Colab verification protocol

Run verification in layers. A successful package import does not prove that a
historical checkpoint or dataset layout is compatible.

## 1. Fresh runtime and package tests

Use a GPU runtime and choose **one** setup method.

### A. Clone after the repository is published

```python
!nvidia-smi
!git clone YOUR_REPOSITORY_URL wound-forecasting
%cd /content/wound-forecasting
!test -f pyproject.toml && echo "Repository root confirmed"
!python -m pip install -U pip
!python -m pip install -e ".[dev,all]"
!python scripts/verify_environment.py --device cuda
!pytest -q
```

Do not continue if cloning or `%cd` fails. The installation output must say
`Obtaining file:///content/wound-forecasting`, not `file:///content`.

### B. Test a local archive before publishing

Upload `wound-forecasting-colab.zip` through Colab's Files panel, then run:

```python
!rm -rf /content/wound-forecasting
!unzip -q /content/wound-forecasting-colab.zip -d /content
%cd /content/wound-forecasting

from pathlib import Path
assert Path("pyproject.toml").is_file(), "Not at the repository root"
assert Path("scripts/verify_environment.py").is_file(), "Verifier is missing"

!python -m pip install -U pip
!python -m pip install -e ".[dev,all]"
!python scripts/verify_environment.py --device cuda
!pytest -q
```

Expected outcome: `Environment verification: PASS` followed by passing tests.
This layer uses synthetic tensors and requires neither data nor checkpoints.

## 2. Artifact identity checks

Mount Drive or authenticate to a private artifact host. Before loading a weight,
confirm its filename, source repository, and recorded model configuration
against `artifacts/checkpoints/manifest.tsv`. For a private archival copy, also
record and compare its byte size outside the public repository:

```python
from google.colab import drive
drive.mount("/content/drive")

from pathlib import Path

checkpoint = Path("/path/to/checkpoint.pth")
print("Filename:", checkpoint.name)
print("Size bytes:", checkpoint.stat().st_size)
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

### LLaMA adapter-only checkpoint

The released wound-specific LLaMA artifact uses format
`wound_forecasting_adapter_delta_v1`; it is not a combined 7B checkpoint.
Construct the model from the exact `Emma02/LVM_ckpts` base and from the
configuration embedded in the adapter package, then load the delta with:

```python
from wound_forecasting.llama_adapter import load_adapter_delta

package, load_result = load_adapter_delta(
    model,
    "/path/to/wound_llama_adapter_delta_v1.pth",
    expected_base_model="Emma02/LVM_ckpts",
)
print("Adapter tensors:", len(package["model"]))
print("Unmodified base tensors:", len(load_result.missing_keys))
```

Missing keys are expected to be unchanged tensors already supplied by the
base. The loader strictly rejects unknown, skipped, or mismatched adapter
tensors. The canonical extraction contains 84 tensors. On 2026-08-24, a
fixed-seed regression using seed `20260824` produced exactly the same generated
token sequence as the private complete checkpoint, with zero differing token
positions. Private VQ tokens and generated reference tokens are retained with
the archival checkpoint and are not part of the public repository.

### River project-only checkpoint

Construct `RiverFlowModel` from `configs/river_final.yaml`, inject the external
VQ-MUSE autoencoder, and load `wound_river_delta_v1.pth` with
`load_river_weights`. The public `RiverVectorFieldRegressor` must expose exactly
102 tensors for the final configuration. VQ-MUSE tensors remain unchanged and
are expected to appear as missing base tensors when the project-only delta is
loaded.

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

All three training commands expose `--smoke-test`, which limits the run to a
minimal training pass while retaining the real construction and checkpoint
paths.

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

Record every hosted artifact's repository, revision/commit, filename, byte
size, and configuration in `artifacts/checkpoints/manifest.tsv`. Temporary hosting for
several months is useful for verification, but the paper should not depend on an
undocumented expiring URL.
