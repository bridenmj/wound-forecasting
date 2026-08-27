"""Explicit runtime boundary for the external VQ-MUSE implementation."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path


def load_vqmuse_autoencoder(source_root: str | Path):
    """Construct River's VQ-MUSE wrapper from an external source checkout."""
    root = Path(source_root).expanduser().resolve()
    candidates = [
        (root / "Shared" / "vq" / "vqvae_muse.py", "Shared.vq.vqvae_muse"),
        (root / "vqvae_muse.py", "vqvae_muse"),
    ]
    for source_file, module_name in candidates:
        if source_file.is_file():
            if str(root) not in sys.path:
                sys.path.insert(0, str(root))
            module = importlib.import_module(module_name)
            return module.VQMuseEncoder(module.get_tokenizer_muse())
    expected = [str(path) for path, _ in candidates]
    raise FileNotFoundError(f"VQ-MUSE source was not found; expected one of {expected}")
