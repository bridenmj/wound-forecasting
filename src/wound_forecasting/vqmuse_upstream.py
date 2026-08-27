"""Explicit runtime boundary for the external VQ-MUSE implementation."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import torch
from torch import Tensor, nn


class VQMuseAutoencoder(nn.Module):
    """Frozen River-facing wrapper around an upstream VQ-MUSE model."""

    def __init__(self, model: nn.Module):
        super().__init__()
        self.net = model.eval()
        for parameter in self.net.parameters():
            parameter.requires_grad = False

    @torch.no_grad()
    def encode(self, images: Tensor) -> Tensor:
        latents, _ = self.net.encode(images)
        return latents

    @torch.no_grad()
    def decode(self, latents: Tensor) -> Tensor:
        return self.net.decode(latents)

    @torch.no_grad()
    def quantize(self, latents: Tensor) -> Tensor:
        quantized, _, _ = self.net.quantize(latents, return_loss=False)
        return quantized


def load_vqmuse_autoencoder(
    source_root: str | Path,
    checkpoint_dir: str | Path,
) -> VQMuseAutoencoder:
    """Load VQ-MUSE from explicit source and checkpoint locations."""
    root = Path(source_root).expanduser().resolve()
    checkpoint = Path(checkpoint_dir).expanduser().resolve()
    source_file = root / "vqvae_muse.py"
    if not source_file.is_file():
        raise FileNotFoundError(f"VQ-MUSE source file is missing: {source_file}")
    if not checkpoint.is_dir():
        raise FileNotFoundError(f"VQ-MUSE checkpoint directory is missing: {checkpoint}")
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    module = importlib.import_module("vqvae_muse")
    model = module.VQGANModel.from_pretrained(str(checkpoint))
    return VQMuseAutoencoder(model)
