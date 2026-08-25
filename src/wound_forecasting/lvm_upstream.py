"""Explicit boundary to the external LVM/LLaMA-Adapter implementation."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path


def load_lvm_components(source_root: str | Path):
    """Import the exact upstream modules required by this project."""
    root = Path(source_root).expanduser().resolve()
    required = [root / "llama" / "llama_adapter.py", root / "vqvae_muse.py"]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"LVM source files are missing: {missing}")
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    adapter_module = importlib.import_module("llama.llama_adapter")
    vq_module = importlib.import_module("vqvae_muse")
    return adapter_module.LLaMA_adapter, vq_module.get_tokenizer_muse


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
