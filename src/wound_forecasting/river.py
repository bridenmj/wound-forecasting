"""Project-specific River checkpoint loading and held-out evaluation."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping
from os import PathLike

import torch


def load_river_weights(
    model,
    checkpoint: str | PathLike[str] | Mapping,
    *,
    map_location: str | torch.device = "cpu",
):
    """Load either the archival checkpoint or released River-only tensors."""
    package = (
        dict(checkpoint)
        if isinstance(checkpoint, Mapping)
        else torch.load(checkpoint, map_location=map_location, weights_only=False)
    )
    state = package.get("model", package)
    if not isinstance(state, Mapping) or not state:
        raise ValueError("River checkpoint contains no model state")
    state = {
        (key.removeprefix("module.")): value
        for key, value in state.items()
        if isinstance(value, torch.Tensor)
    }
    model_keys = set(model.state_dict())
    unknown = sorted(set(state) - model_keys)
    if unknown:
        raise KeyError(f"River checkpoint has unknown tensors: {unknown}")
    result = model.load_state_dict(state, strict=False)
    not_loaded = sorted(set(state) & set(result.missing_keys))
    if not_loaded:
        raise RuntimeError(f"River tensors were not loaded: {not_loaded}")
    loaded = model.state_dict()
    mismatched = [
        key
        for key, value in state.items()
        if not torch.equal(loaded[key].detach().cpu(), value.detach().cpu())
    ]
    if mismatched:
        raise RuntimeError(f"Loaded River tensors differ: {mismatched}")
    return package, result


@torch.inference_mode()
def generate_river_pools(
    model,
    loader,
    *,
    device: str | torch.device,
    steps: int = 100,
    context_frames: int = 4,
    past_horizon: int = -1,
):
    """Generate aligned pools from padded variable-length sequences."""
    fake, real = defaultdict(list), defaultdict(list)
    model.eval()
    for data, lengths in loader:
        data = data.to(device)
        for length in sorted({int(value) for value in lengths}):
            indices = (lengths == length).nonzero(as_tuple=True)[0]
            if length <= context_frames or indices.numel() == 0:
                continue
            sequence = data.index_select(0, indices.to(data.device))[:, :length]
            future_count = length - context_frames
            generated = model.generate_frames(
                observations=sequence[:, :context_frames],
                num_frames=future_count,
                steps=steps,
                past_horizon=past_horizon,
                verbose=False,
            )
            predicted = generated[:, context_frames:].detach().cpu().clamp(0, 1)
            targets = sequence[:, context_frames:].detach().cpu().clamp(0, 1)
            for offset in range(future_count):
                horizon = offset + 1
                fake[horizon].append(predicted[:, offset])
                real[horizon].append(targets[:, offset])
    return (
        {key: torch.cat(value) for key, value in fake.items()},
        {key: torch.cat(value) for key, value in real.items()},
    )
