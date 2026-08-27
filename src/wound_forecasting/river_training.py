"""Optimization and checkpoint operations for the wound-specific River CFM."""

from __future__ import annotations

from collections.abc import Mapping

import torch
from torch import Tensor

from .river import RiverFlowModel, river_flow_matching_loss


def train_river_step(
    model: RiverFlowModel,
    observations: Tensor,
    optimizer: torch.optim.Optimizer,
    *,
    gradient_accumulation_steps: int = 1,
) -> float:
    """Run one gradient-accumulation microstep using the final CFM objective."""
    if gradient_accumulation_steps < 1:
        raise ValueError("gradient_accumulation_steps must be positive")
    output = model(observations)
    loss = river_flow_matching_loss(output)
    (loss / gradient_accumulation_steps).backward()
    return float(loss.detach().item())


@torch.inference_mode()
def validate_river_loss(
    model: RiverFlowModel,
    batches,
    *,
    device: str | torch.device,
) -> float:
    """Compute mean pointwise flow-matching loss over validation trajectories."""
    model.eval()
    losses = []
    for observations in batches:
        if isinstance(observations, (tuple, list)):
            observations = observations[0]
        losses.append(float(river_flow_matching_loss(model(observations.to(device))).item()))
    if not losses:
        raise RuntimeError("No River validation batches were evaluated")
    return sum(losses) / len(losses)


def river_checkpoint_package(
    model: RiverFlowModel,
    *,
    step: int,
    config: Mapping,
    validation_loss: float | None = None,
) -> dict:
    """Save only project-trained River tensors, excluding external VQ-MUSE."""
    state = {
        name: value.detach().cpu().clone()
        for name, value in model.state_dict().items()
        if name.startswith("vector_field_regressor.")
    }
    return {
        "format": "wound_river_delta_v1",
        "model": state,
        "step": int(step),
        "validation_loss": validation_loss,
        "config": dict(config),
    }
