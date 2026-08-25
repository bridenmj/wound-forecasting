"""Checkpoint loading and bucketed inference for variable-context DyneODE."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping
from pathlib import Path

import torch
from torch import Tensor

from .dyneode import ContextConditionedODEFunc, collapse_broadcast_w
from .dyneode_training import decode_stylegan_w, forecast_task


def load_dyneode_checkpoint(
    checkpoint: str | Path | Mapping,
    *,
    device: str | torch.device = "cpu",
) -> tuple[ContextConditionedODEFunc, dict]:
    """Construct and strict-load the final self-describing DyneODE model."""
    if isinstance(checkpoint, Mapping):
        package = dict(checkpoint)
    else:
        package = torch.load(checkpoint, map_location="cpu", weights_only=False)
    required = {
        "odefunc",
        "hidden_dim",
        "depth",
        "context_encoder_type",
        "context_hidden_dim",
        "time_scale",
    }
    missing = required - set(package)
    if missing:
        raise ValueError(f"DyneODE checkpoint is missing: {sorted(missing)}")
    state = package["odefunc"]
    if not isinstance(state, Mapping) or not state:
        raise TypeError("Checkpoint odefunc state must be a non-empty mapping")
    dim = int(package.get("style_dim", package.get("dim", 512)))
    model = ContextConditionedODEFunc(
        dim=dim,
        hidden_dim=int(package["hidden_dim"]),
        depth=int(package["depth"]),
        context_encoder_type=str(package["context_encoder_type"]),
        context_hidden_dim=int(package["context_hidden_dim"]),
    )
    model.load_state_dict(state, strict=True)
    model.to(device).eval()
    return model, package


@torch.inference_mode()
def predict_trajectory(
    odefunc: ContextConditionedODEFunc,
    latents: Tensor,
    times: Tensor,
    *,
    context_size: int,
    time_scale: float,
    odeint_fn: Callable,
) -> Tensor:
    """Forecast all frames after a prefix of observed context frames."""
    latents = collapse_broadcast_w(latents)
    if len(latents) <= context_size:
        raise ValueError("Trajectory must contain at least one future frame")
    prediction, _ = forecast_task(
        odefunc,
        latents,
        times,
        context_size=context_size,
        anchor_index=context_size - 1,
        time_scale=time_scale,
        odeint_fn=odeint_fn,
    )
    return prediction[1:]


@torch.inference_mode()
def build_dyneode_image_pools(
    odefunc: ContextConditionedODEFunc,
    batches: Iterable[Mapping[str, object]],
    *,
    generator,
    odeint_fn: Callable,
    device: str | torch.device,
    context_size: int = 4,
    time_scale: float = 21.0,
) -> tuple[dict[int, Tensor], dict[int, Tensor]]:
    """Generate one aligned image prediction per unique target and horizon."""
    fake: defaultdict[int, list[Tensor]] = defaultdict(list)
    real: defaultdict[int, list[Tensor]] = defaultdict(list)
    odefunc.eval()
    generator.eval()
    for batch in batches:
        latents = batch["latents"]
        times = batch["t_steps"]
        if not isinstance(latents, Tensor) or not isinstance(times, Tensor):
            raise TypeError("Batch latents and t_steps must be tensors")
        if latents.ndim == 3:
            if len(latents) != 1:
                raise ValueError("Evaluation requires batch_size=1")
            latents, times = latents[0], times[0]
        latents = collapse_broadcast_w(latents.to(device))
        times = times.to(device)
        if len(latents) <= context_size:
            continue
        predicted = predict_trajectory(
            odefunc,
            latents,
            times,
            context_size=context_size,
            time_scale=time_scale,
            odeint_fn=odeint_fn,
        )
        targets = latents[context_size:]
        predicted_images = (decode_stylegan_w(generator, predicted) + 1) / 2
        target_images = (decode_stylegan_w(generator, targets) + 1) / 2
        for index, (prediction, target) in enumerate(
            zip(predicted_images.cpu(), target_images.cpu(), strict=True), start=1
        ):
            fake[index].append(prediction)
            real[index].append(target)
    if not fake:
        raise RuntimeError("No eligible DyneODE forecasts were generated")
    return (
        {horizon: torch.stack(images) for horizon, images in sorted(fake.items())},
        {horizon: torch.stack(images) for horizon, images in sorted(real.items())},
    )
