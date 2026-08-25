"""Training-loop helpers for the wound-specific LLaMA adapter."""

from __future__ import annotations

from collections.abc import Mapping

import torch

from .llama_adapter import ADAPTER_DELTA_FORMAT


def trainable_parameter_names(model) -> list[str]:
    """Return the adapter tensors selected by the upstream architecture."""
    names = [
        name for name, parameter in model.named_parameters() if parameter.requires_grad
    ]
    if not names:
        raise ValueError("Model has no trainable adapter parameters")
    return names


def build_adapter_delta(
    model,
    *,
    base_model: str,
    architecture: Mapping,
    epoch: int | None = None,
) -> dict:
    """Create a portable adapter-only checkpoint from trainable tensors."""
    state = model.state_dict()
    names = trainable_parameter_names(model)
    package = {
        "format": ADAPTER_DELTA_FORMAT,
        "config": {"base_model": base_model, **dict(architecture)},
        "model": {name: state[name].detach().cpu().clone() for name in names},
    }
    if epoch is not None:
        package["epoch"] = int(epoch)
    return package


def save_adapter_delta(path, model, **metadata) -> None:
    """Save the project-trained tensors without redistributing base weights."""
    torch.save(build_adapter_delta(model, **metadata), path)
