"""Explicit boundary to the external LVM/LLaMA-Adapter implementation."""

from __future__ import annotations

import importlib
import sys
from functools import partial
from pathlib import Path

from .llama_text_adapter import WoundLLaMAAdapter


def load_vqmuse_checkpoint(vq_module, checkpoint_dir: str | Path):
    """Load VQ-MUSE without relying on the upstream hardcoded path."""
    checkpoint = Path(checkpoint_dir).expanduser().resolve()
    if not checkpoint.is_dir():
        raise FileNotFoundError(
            f"VQ-MUSE checkpoint directory is missing: {checkpoint}"
        )
    return vq_module.VQGANModel.from_pretrained(str(checkpoint))


def load_lvm_components(
    source_root: str | Path,
    vq_source_root: str | Path | None = None,
):
    """Combine the wound adapter with separately installed upstream sources."""
    root = Path(source_root).expanduser().resolve()
    vq_root = (
        root
        if vq_source_root is None
        else Path(vq_source_root).expanduser().resolve()
    )
    required = [
        root / "llama" / "llama.py",
        root / "llama" / "utils.py",
        vq_root / "vqvae_muse.py",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"LVM source files are missing: {missing}")
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    if str(vq_root) not in sys.path:
        sys.path.insert(0, str(vq_root))
    vq_module = importlib.import_module("vqvae_muse")

    adapter_class = partial(WoundLLaMAAdapter, upstream_source_root=root)
    return adapter_class, partial(load_vqmuse_checkpoint, vq_module)


def construct_wound_llama(
    adapter_class,
    *,
    llama_checkpoint_dir: str | Path,
    maximum_sequence_length: int = 2048,
    maximum_batch_size: int = 20,
    image_adapter_length: int = 32,
    text_adapter_length: int = 32,
    query_layer: int = 30,
    text_in_layers: int = 4,
    use_text: bool = True,
    phase: str = "pretrain",
):
    """Construct the architecture used for the released wound adapter."""
    return adapter_class(
        str(Path(llama_checkpoint_dir).expanduser().resolve()),
        max_seq_len=maximum_sequence_length,
        img_adapter_len=image_adapter_length,
        text_adapter_len=text_adapter_length,
        use_text=use_text,
        max_batch_size=maximum_batch_size,
        query_layer=query_layer,
        text_in_layers=text_in_layers,
        phase=phase,
    )
