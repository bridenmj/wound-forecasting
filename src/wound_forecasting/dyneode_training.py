"""Training and validation operations for variable-context DyneODE."""

from __future__ import annotations

import random
from collections.abc import Callable, Iterable, Mapping

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .dyneode import ContextConditionedODEFunc, collapse_broadcast_w, conditioned_odeint


def context_forecast_tasks(
    trajectory_length: int,
    min_context_size: int,
    max_context_size: int,
) -> list[tuple[int, int]]:
    """Return every valid ``(context_size, anchor_index)`` task."""
    if trajectory_length < 2:
        return []
    if min_context_size < 1 or max_context_size < min_context_size:
        raise ValueError("Invalid context-size range")
    return [
        (context_size, anchor_index)
        for context_size in range(
            min_context_size,
            min(max_context_size, trajectory_length - 1) + 1,
        )
        for anchor_index in range(context_size - 1, trajectory_length - 1)
    ]


def decode_stylegan_w(generator: nn.Module, latents: Tensor) -> Tensor:
    """Decode Single-W vectors through the wound StyleGAN generator."""
    if latents.ndim != 2:
        raise ValueError(f"Expected [N,512] latents, got {tuple(latents.shape)}")
    expanded = latents.unsqueeze(1).expand(-1, generator.n_latent, -1)
    images = generator(
        [expanded],
        input_is_latent=True,
        randomize_noise=False,
        return_latents=False,
    )[0]
    return images.clamp(-1, 1)


def forecast_task(
    odefunc: ContextConditionedODEFunc,
    latents: Tensor,
    times: Tensor,
    *,
    context_size: int,
    anchor_index: int,
    time_scale: float,
    odeint_fn: Callable,
) -> tuple[Tensor, Tensor]:
    """Forecast one suffix using the context window ending at an anchor."""
    if time_scale <= 0:
        raise ValueError("time_scale must be positive")
    context_start = anchor_index - context_size + 1
    if context_start < 0 or anchor_index >= len(latents) - 1:
        raise ValueError("Invalid context/anchor combination")
    normalized_times = times / time_scale
    return conditioned_odeint(
        odefunc=odefunc,
        context_latents=latents[context_start : anchor_index + 1],
        context_times=normalized_times[context_start : anchor_index + 1],
        forecast_times=normalized_times[anchor_index:],
        odeint_fn=odeint_fn,
    )


def latent_forecast_losses(prediction: Tensor, target: Tensor) -> dict[str, Tensor]:
    """Compute the pair and weighted-suffix objective used in training."""
    if prediction.shape != target.shape or len(prediction) < 2:
        raise ValueError("Prediction and target must align and include a future")
    per_time = (prediction[1:] - target[1:]).square().mean(dim=1)
    weights = torch.linspace(0.7, 1.2, steps=len(per_time), device=prediction.device)
    suffix = (weights * per_time).mean()
    pair = F.mse_loss(prediction[1], target[1])
    return {"pair_loss": pair, "suffix_loss": suffix, "loss": 0.5 * (pair + suffix)}


def train_trajectory_step(
    *,
    odefunc: ContextConditionedODEFunc,
    optimizer: torch.optim.Optimizer,
    latents: Tensor,
    times: Tensor,
    odeint_fn: Callable,
    generator: nn.Module,
    perceptual_metric: nn.Module,
    min_context_size: int = 1,
    max_context_size: int = 7,
    time_scale: float = 21.0,
    latent_loss_weight: float = 0.5,
    perceptual_loss_weight: float = 0.5,
    latent_loss_scale: float = 7.0,
    perceptual_loss_scale: float = 0.45,
    rng: random.Random | None = None,
) -> dict[str, float]:
    """Optimize one randomly selected variable-context task."""
    latents = collapse_broadcast_w(latents)
    tasks = context_forecast_tasks(len(latents), min_context_size, max_context_size)
    if not tasks:
        raise ValueError("Trajectory has no valid forecast task")
    context_size, anchor = (rng or random).choice(tasks)
    prediction, _ = forecast_task(
        odefunc,
        latents,
        times,
        context_size=context_size,
        anchor_index=anchor,
        time_scale=time_scale,
        odeint_fn=odeint_fn,
    )
    target = latents[anchor:]
    losses = latent_forecast_losses(prediction, target)

    with torch.no_grad():
        target_images = decode_stylegan_w(generator, target[1:])
    predicted_images = decode_stylegan_w(generator, prediction[1:])
    perceptual = perceptual_metric(
        predicted_images, target_images, normalize=False
    ).mean()
    combined = (
        latent_loss_weight * losses["loss"] / latent_loss_scale
        + perceptual_loss_weight * perceptual / perceptual_loss_scale
    )
    optimizer.zero_grad(set_to_none=True)
    combined.backward()
    gradient_norm = torch.nn.utils.clip_grad_norm_(
        odefunc.parameters(), max_norm=float("inf")
    )
    optimizer.step()
    return {
        **{name: float(value.detach().item()) for name, value in losses.items()},
        "lpips_loss": float(perceptual.detach().item()),
        "combined_loss": float(combined.detach().item()),
        "gradient_norm": float(gradient_norm.detach().item()),
        "context_size": int(context_size),
    }


@torch.inference_mode()
def validate_ode(
    odefunc: ContextConditionedODEFunc,
    batches: Iterable[Mapping[str, object]],
    *,
    device: str | torch.device,
    odeint_fn: Callable,
    time_scale: float = 21.0,
    min_context_size: int = 1,
    max_context_size: int = 7,
    generator: nn.Module | None = None,
    perceptual_metric: nn.Module | None = None,
    latent_loss_scale: float = 7.0,
    perceptual_loss_scale: float = 0.45,
    latent_loss_weight: float = 0.5,
    perceptual_loss_weight: float = 0.5,
) -> dict[str, float]:
    """Evaluate every valid context/anchor task deterministically."""
    odefunc.eval()
    use_perceptual = generator is not None and perceptual_metric is not None
    rows: list[dict[str, float]] = []
    for batch in batches:
        latents = batch["latents"]
        times = batch["t_steps"]
        if not isinstance(latents, Tensor) or not isinstance(times, Tensor):
            raise TypeError("Batch latents and t_steps must be tensors")
        if latents.ndim == 3:
            if len(latents) != 1:
                raise ValueError("Variable-length validation requires batch_size=1")
            latents, times = latents[0], times[0]
        latents = collapse_broadcast_w(latents.to(device))
        times = times.to(device)
        for context_size, anchor in context_forecast_tasks(
            len(latents), min_context_size, max_context_size
        ):
            prediction, _ = forecast_task(
                odefunc,
                latents,
                times,
                context_size=context_size,
                anchor_index=anchor,
                time_scale=time_scale,
                odeint_fn=odeint_fn,
            )
            target = latents[anchor:]
            losses = latent_forecast_losses(prediction, target)
            row = {name: float(value.item()) for name, value in losses.items()}
            if use_perceptual:
                target_images = decode_stylegan_w(generator, target[1:])
                predicted_images = decode_stylegan_w(generator, prediction[1:])
                stationary_images = decode_stylegan_w(
                    generator, target[:1].expand_as(target[1:])
                )
                row["lpips_loss"] = float(
                    perceptual_metric(predicted_images, target_images, normalize=False)
                    .mean()
                    .item()
                )
                row["stationary_lpips"] = float(
                    perceptual_metric(stationary_images, target_images, normalize=False)
                    .mean()
                    .item()
                )
            rows.append(row)
    if not rows:
        raise RuntimeError("No valid validation tasks were evaluated")
    metrics = {
        name: sum(row[name] for row in rows) / len(rows)
        for name in ("loss", "pair_loss", "suffix_loss")
    }
    metrics["tasks"] = len(rows)
    if use_perceptual:
        metrics["lpips_loss"] = sum(row["lpips_loss"] for row in rows) / len(rows)
        metrics["stationary_lpips"] = sum(
            row["stationary_lpips"] for row in rows
        ) / len(rows)
        metrics["lpips_improvement_pct"] = (
            100.0
            * (metrics["stationary_lpips"] - metrics["lpips_loss"])
            / max(metrics["stationary_lpips"], 1e-8)
        )
        metrics["combined_loss"] = (
            latent_loss_weight * metrics["loss"] / latent_loss_scale
            + perceptual_loss_weight * metrics["lpips_loss"] / perceptual_loss_scale
        )
    return metrics


def checkpoint_package(
    odefunc: ContextConditionedODEFunc,
    *,
    epoch: int,
    global_step: int,
    validation_metrics: Mapping[str, float],
    config: Mapping[str, object],
) -> dict:
    """Create the self-describing checkpoint format used by the final run."""
    return {
        "epoch": epoch,
        "global_step": global_step,
        "odefunc": {
            key: value.detach().cpu().clone()
            for key, value in odefunc.state_dict().items()
        },
        "val_metrics": dict(validation_metrics),
        **dict(config),
    }
