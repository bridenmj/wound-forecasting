# TMIH source snapshot audit

Audit date: 2026-08-23

Source bundle: `tmih_all_source_snapshots.zip`

This was a read-only audit. No Google Drive source files or archive contents were modified.

## Executive finding

The bundle is a useful source-code baseline, but it is not yet the frozen “as used for the paper” snapshot. River’s core implementation and final YAML are substantially present. LLaMA-Adapter contains the upstream/custom adapter and VQ code but not the final experiment notebook. DyneODE contains upstream StyleGAN/DynODE support and analysis outputs but not the variable-context model implementation used in the paper.

None of the three archives contains a notebook. The corrected shared metric implementation is also absent.

## LLaMA-Adapter

### Present

- Adapter implementation under `llama/`
- Dataset implementation at `data/dataset.py`
- VQ-MUSE-related implementation at `vqvae_muse.py` and `vqvae/`
- Training entry points (`main_finetune.py`, `engine_finetune.py`)
- YAML dataset configurations
- `requirements.txt`
- `val_manifest.json`

### Missing from the paper snapshot

- Final notebook: `(Main)_hybrid_text_llama_adapter_nov_30.ipynb`
- Corrected shared metrics module
- Final manifest/prompt-generation helpers used by the notebook, unless entirely embedded in that notebook
- Exact final adapter checkpoint metadata and checksum
- Final compact metric results in a portable text format
- Canonical publication figures and exact seed-29 qualitative artifact

### Large-artifact note

The excluded manifest lists numerous roughly 13 GB training checkpoints. These should not be downloaded or committed wholesale. Only the exact checkpoint used for final evaluation should be identified, checksummed, and hosted externally.

## DyneODE

### Present

- Upstream StyleGAN2 implementation under `code/stylegan2/`
- Legacy/upstream evaluation support under `evaluation/`
- Upstream DynODE README and license
- PCA, UMAP, LPIPS, and predictability analysis summaries

### Missing from the paper snapshot

- Final notebook: `DyneODE_III_variable_context.ipynb`
- `ContextConditionedODEfunc`
- GRU variable-context encoder implementation
- `conditioned_odeint`
- Final training/evaluation dataset construction
- Corrected KID and targetwise PSNR/SSIM implementation
- Exact final variable-context checkpoint metadata and checksum
- Final per-pig and combined metrics in portable text format

The absence of the final model definitions confirms that the notebook is currently the canonical implementation rather than this Drive source tree.

## River

### Present

- Dataset/HDF5 implementation under `dataset/`
- River model and vector-field regressor under `model/`
- Training and evaluator implementations
- Sweep configuration and per-run sweep YAML files
- Final training configuration: `train_configs/driven_8_final.yaml`
- Environment file and upstream dependencies under `ldm/`

### Confirmed final configuration characteristics

`train_configs/driven_8_final.yaml` specifies four conditioning frames, one-frame training targets, four-frame evaluation generation, VQ-MUSE, 40,000 optimizer steps, gradient accumulation of four, and the final vector-field dimensions.

### Missing or externally coupled

- Final notebook: `River0.ipynb`
- `Shared.vq.vqvae_muse`, imported by `model/model.py`
- Any other modules under `/content/drive/MyDrive/Heals_Winter_24/Shared`
- Corrected shared metrics module
- Manifest-to-HDF5 paper evaluation helpers if embedded only in the notebook
- Exact final checkpoint metadata/checksum for `runs/davinci_river_vqmuse_run-driven-8-40k_final_run/checkpoints/latest.pth`
- Final per-pig and combined metrics in portable text format

The final River checkpoint is about 750 MB and should be externally hosted rather than committed to ordinary Git.

## Portability issues

- Several files contain absolute `/content` or Google Drive paths.
- River’s model mutates `sys.path` and imports `Shared.vq` from outside its project root.
- River configurations use `/content/data` directly.
- LLaMA dataset YAML files contain absolute Google Drive paths.
- The source bundle does not capture the final evaluation functions that standardized KID and targetwise PSNR/SSIM across models.

These should be documented first and parameterized only after the frozen paper snapshot is preserved.

## Exact next collection

The next archive should stay small and contain only:

1. `(Main)_hybrid_text_llama_adapter_nov_30.ipynb`
2. `DyneODE_III_variable_context.ipynb`
3. `River0.ipynb`
4. The current corrected `metrics.py`
5. `create_test.py` or its current manifest helper equivalent
6. The complete `/content/drive/MyDrive/Heals_Winter_24/Shared/vq/` source directory used by River
7. Prompt JSONL and manifest JSON files required to reconstruct splits, provided they contain no restricted raw data
8. Final metric results exported as JSON, CSV, or text
9. Final paper and supplemental figures
10. A checkpoint manifest listing only the three canonical checkpoint paths, sizes, hashes, and model/config associations

Do not include raw images, latent inversions, HDF5 shards, W&B directories, or the model checkpoints themselves in this second source archive.

## Repository decision

No repository reorganization should occur yet. First combine this source baseline with the exact-next-collection files into a frozen `paper_snapshot/`. Once that snapshot is complete, reusable modules can be extracted into `src/` without losing provenance.

## Shared VQ follow-up archive

Archive inspected: `vq-20260824T045138Z-1-001.zip`

This archive supplies River's previously missing `Shared.vq` dependency. The required source is present:

- `vq/__init__.py`
- `vq/vqvae_muse.py`
- `vq/vqvae/__init__.py`
- `vq/vqvae/logging.py`
- `vq/vqvae/modeling_utils.py`
- `vq/vqvae/Emma02/vqvae_ckpts/config.json`
- `vq/vqvae/Emma02/vqvae_ckpts/README.md`

For the frozen repository snapshot, this directory should be placed at `Shared/vq/` so that River's current import `from Shared.vq.vqvae_muse ...` resolves without changing the paper-era code.

The archive also contains a 585,077,729-byte `pytorch_model.bin` and a second 585,077,729-byte Git LFS object inside a nested `.git` directory. The nested `.git`, caches, `.DS_Store`, and duplicate LFS object must not enter the new repository. The model binary is required by `get_tokenizer_muse()` at runtime, but it should be represented in Git only by external-download metadata and a checksum. Its SHA-256 digest is `fed960b9b88968dcaab28fd0cb28b786693b83b21fa93b1a1ad1177b3f7e9fd1`.

The source declares additional runtime dependencies on `accelerate`, `huggingface_hub`, `requests`, PyTorch, NumPy, and tqdm.
