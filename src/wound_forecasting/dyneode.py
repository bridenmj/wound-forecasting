"""Variable-context DyneODE architecture used in the final experiment."""

from collections.abc import Callable

import torch
from torch import Tensor, nn


def collapse_broadcast_w(latents: Tensor, atol: float = 1e-5, rtol: float = 1e-5) -> Tensor:
    """Convert ``[T, layers, 512]`` broadcast-W storage to ``[T, 512]``."""
    if latents.ndim == 2 and latents.shape[-1] == 512:
        return latents
    if latents.ndim < 3 or latents.shape[-1] != 512:
        raise ValueError(f"Expected [T, 512] or [T, L, 512], got {tuple(latents.shape)}")
    first_w = latents[..., 0, :]
    expected = first_w.unsqueeze(-2).expand_as(latents)
    if not torch.allclose(latents, expected, atol=atol, rtol=rtol):
        difference = (latents - expected).abs().max().item()
        raise ValueError(f"Latent rows are not broadcast W copies; max difference={difference:.6g}")
    return first_w


def estimate_context_slope(latents: Tensor, times: Tensor, eps: float = 1e-8) -> Tensor:
    """Estimate an irregular-time linear slope for a context window."""
    _validate_context(latents, times)
    if len(times) == 1:
        return torch.zeros_like(latents[0])
    centered_times = times - times.mean()
    centered_latents = latents - latents.mean(dim=0, keepdim=True)
    return (centered_times[:, None] * centered_latents).sum(dim=0) / (
        centered_times.square().sum().clamp_min(eps)
    )


def _validate_context(latents: Tensor, times: Tensor) -> None:
    if latents.ndim != 2 or times.ndim != 1 or len(latents) != len(times):
        raise ValueError("Expected aligned context latents [C,D] and times [C]")
    if len(times) < 1:
        raise ValueError("At least one context observation is required")
    if len(times) > 1 and torch.any(times[1:] <= times[:-1]):
        raise ValueError("Context times must be strictly increasing")


class TimeAwareGRUContextEncoder(nn.Module):
    """Encode latent observations together with absolute and elapsed time."""

    def __init__(self, dim: int = 512, hidden_dim: int = 128):
        super().__init__()
        self.input_projection = nn.Sequential(
            nn.Linear(dim + 2, hidden_dim), nn.LayerNorm(hidden_dim), nn.LeakyReLU(0.2)
        )
        self.gru = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.output_projection = nn.Linear(hidden_dim, dim)

    def forward(self, latents: Tensor, times: Tensor) -> Tensor:
        _validate_context(latents, times)
        delta = torch.zeros_like(times)
        delta[1:] = times[1:] - times[:-1]
        values = torch.cat([latents, times[:, None], delta[:, None]], dim=-1)
        _, hidden = self.gru(self.input_projection(values).unsqueeze(0))
        return self.output_projection(hidden[-1]).squeeze(0)


class ContextConditionedODEFunc(nn.Module):
    """Time- and variable-context-conditioned latent vector field."""

    def __init__(
        self,
        dim: int = 512,
        hidden_dim: int = 512,
        depth: int = 3,
        context_encoder_type: str = "gru",
        context_hidden_dim: int = 128,
    ):
        super().__init__()
        if depth < 1:
            raise ValueError("depth must be at least 1")
        if context_encoder_type not in {"gru", "slope"}:
            raise ValueError("context_encoder_type must be 'gru' or 'slope'")
        self.context_encoder_type = context_encoder_type
        self.context_encoder = (
            TimeAwareGRUContextEncoder(dim, context_hidden_dim)
            if context_encoder_type == "gru"
            else None
        )
        layers: list[nn.Module] = []
        input_dim = 2 * dim + 1
        for _ in range(depth - 1):
            layers.extend([nn.Linear(input_dim, hidden_dim), nn.LeakyReLU(0.2)])
            input_dim = hidden_dim
        layers.append(nn.Linear(input_dim, dim))
        self.model = nn.Sequential(*layers)

    def encode_context(self, latents: Tensor, times: Tensor) -> Tensor:
        if self.context_encoder_type == "gru":
            assert self.context_encoder is not None
            return self.context_encoder(latents, times)
        return estimate_context_slope(latents, times)

    def forward(self, time: Tensor, state: Tensor, context: Tensor) -> Tensor:
        if state.shape != context.shape:
            raise ValueError("state and context must have identical shapes")
        time_feature = torch.ones_like(state[..., :1]) * time
        return self.model(torch.cat([state, context, time_feature], dim=-1))


def conditioned_odeint(
    odefunc: ContextConditionedODEFunc,
    context_latents: Tensor,
    context_times: Tensor,
    forecast_times: Tensor,
    odeint_fn: Callable,
    method: str = "rk4",
    atol: float = 1e-3,
    rtol: float = 1e-3,
) -> tuple[Tensor, Tensor]:
    """Encode context once and integrate from the final observed state."""
    if torch.any(forecast_times[1:] <= forecast_times[:-1]):
        raise ValueError("Forecast times must be strictly increasing")
    if not torch.allclose(context_times[-1], forecast_times[0], atol=1e-7, rtol=1e-6):
        raise ValueError("Final context time must equal the first forecast time")
    context = odefunc.encode_context(context_latents, context_times)
    trajectory = odeint_fn(
        lambda time, state: odefunc(time, state, context),
        context_latents[-1],
        forecast_times,
        method=method,
        atol=atol,
        rtol=rtol,
    )
    return trajectory, context

