"""Loading boundary for the external StyleGAN2 implementation."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import torch


def load_wound_stylegan(
    checkpoint_path: str | Path,
    *,
    source_root: str | Path,
    device: str | torch.device = "cpu",
    image_size: int = 256,
    style_dim: int = 512,
    mapping_network_depth: int = 8,
    channel_multiplier: int = 2,
):
    """Construct the upstream generator and strict-load the wound weights.

    ``source_root`` must make ``stylegan2.model.Generator`` importable. The
    external implementation is deliberately not duplicated in this package.
    """
    source = Path(source_root).expanduser().resolve()
    if not source.is_dir():
        raise FileNotFoundError(f"StyleGAN source directory not found: {source}")
    checkpoint_path = Path(checkpoint_path).expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"StyleGAN checkpoint not found: {checkpoint_path}")
    sys.path.insert(0, str(source))
    try:
        module = importlib.import_module("stylegan2.model")
    finally:
        if sys.path[0] == str(source):
            sys.path.pop(0)
    generator = module.Generator(
        image_size,
        style_dim,
        mapping_network_depth,
        channel_multiplier=channel_multiplier,
    )
    package = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(package, dict) or "g_ema" not in package:
        raise ValueError("StyleGAN checkpoint must contain the 'g_ema' state")
    generator.load_state_dict(package["g_ema"], strict=True)
    generator.to(device).eval()
    for parameter in generator.parameters():
        parameter.requires_grad_(False)
    return generator, package
