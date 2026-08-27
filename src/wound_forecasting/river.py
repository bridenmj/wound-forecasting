"""Project-modified River CFM architecture, loading, and evaluation.

Only the external VQ-MUSE autoencoder is injected at runtime.  The
self-conditioned vector-field regressor and autoregressive flow-matching
logic trained for the wound experiments live in this package.
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Mapping
from os import PathLike

import torch
import torch.nn.functional as F
from torch import Tensor, nn


def timestamp_embedding(
    timesteps: Tensor,
    dim: int,
    *,
    scale: float = 200,
    max_period: float = 10_000,
) -> Tensor:
    """Create the sinusoidal time embedding used by the final River model."""
    half = dim // 2
    frequencies = torch.exp(
        -math.log(max_period)
        * torch.arange(half, dtype=torch.float32, device=timesteps.device)
        / half
    )
    arguments = scale * timesteps[:, None].float() * frequencies[None]
    embedding = torch.cat([torch.cos(arguments), torch.sin(arguments)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


class LearnedImagePositionEmbedding(nn.Module):
    """Checkpoint-compatible learned 2D positional embedding."""

    def __init__(self, num_pos_feats: int = 256):
        super().__init__()
        self.row_embed = nn.Embedding(512, num_pos_feats)
        self.col_embed = nn.Embedding(512, num_pos_feats)
        nn.init.uniform_(self.row_embed.weight, a=-1.0, b=1.0)
        nn.init.uniform_(self.col_embed.weight, a=-1.0, b=1.0)

    def forward(self, values: Tensor) -> Tensor:
        height, width = values.shape[-2:]
        columns = self.col_embed(torch.arange(width, device=values.device))
        rows = self.row_embed(torch.arange(height, device=values.device))
        position = torch.cat(
            [
                columns.unsqueeze(0).repeat(height, 1, 1),
                rows.unsqueeze(1).repeat(1, width, 1),
            ],
            dim=-1,
        )
        return position.permute(2, 0, 1).unsqueeze(0).repeat(len(values), 1, 1, 1)


class RiverVectorFieldRegressor(nn.Module):
    """Self-conditioned transformer vector field used by the final River run."""

    def __init__(
        self,
        *,
        depth: int,
        mid_depth: int,
        state_size: int,
        state_res: tuple[int, int],
        inner_dim: int,
        out_norm: str = "ln",
        dropout: float = 0.05,
        reference: bool = True,
    ):
        super().__init__()
        self.state_size = state_size
        self.state_height, self.state_width = state_res
        self.inner_dim = inner_dim
        self.reference = reference
        self.position_encoding = LearnedImagePositionEmbedding(inner_dim // 2)
        input_groups = 4 if reference else 3
        self.project_in = nn.Sequential(
            _ImageToTokens(),
            nn.Linear(input_groups * state_size, inner_dim)
        )
        self.time_projection = nn.Sequential(
            nn.Linear(1, 256), nn.ReLU(), nn.Linear(256, inner_dim)
        )

        def transformer_layer() -> nn.TransformerEncoderLayer:
            return nn.TransformerEncoderLayer(
                d_model=inner_dim,
                nhead=8,
                dim_feedforward=4 * inner_dim,
                dropout=dropout,
                activation="gelu",
                norm_first=True,
                batch_first=True,
            )

        self.in_blocks = nn.ModuleList([transformer_layer() for _ in range(depth)])
        self.mid_blocks = nn.Sequential(
            *[transformer_layer() for _ in range(mid_depth)]
        )
        self.out_blocks = nn.ModuleList(
            [
                nn.ModuleList([nn.Linear(2 * inner_dim, inner_dim), transformer_layer()])
                for _ in range(depth)
            ]
        )
        if out_norm == "ln":
            self.project_out = nn.Sequential(
                nn.Linear(inner_dim, inner_dim),
                nn.GELU(),
                nn.LayerNorm(inner_dim),
                _TokensToImage(self.state_height),
                nn.Conv2d(inner_dim, state_size, kernel_size=3, padding=1),
            )
        elif out_norm == "bn":
            self.project_out = nn.Sequential(
                nn.Linear(inner_dim, inner_dim),
                _TokensToImage(self.state_height),
                nn.GELU(),
                nn.BatchNorm2d(inner_dim),
                nn.Conv2d(inner_dim, state_size, kernel_size=3, padding=1),
            )
        else:
            raise ValueError(f"Unsupported output normalization: {out_norm}")

    def forward(
        self,
        input_latents: Tensor,
        reference_latents: Tensor,
        conditioning_latents: Tensor,
        self_cond_latents: Tensor,
        index_distances: Tensor,
        timestamps: Tensor,
    ) -> Tensor:
        time_token = timestamp_embedding(timestamps, self.inner_dim).unsqueeze(1)
        position = self.position_encoding(input_latents)
        position = position.flatten(2).transpose(1, 2)
        distance = self.time_projection(torch.log(index_distances).unsqueeze(1)).unsqueeze(1)
        groups = [input_latents, conditioning_latents, self_cond_latents]
        if self.reference:
            groups.insert(1, reference_latents)
        values = self.project_in(torch.cat(groups, dim=1)) + position + distance
        values = torch.cat([time_token, values], dim=1)
        shortcuts = []
        for block in self.in_blocks:
            values = block(values)
            shortcuts.append(values.clone())
        values = self.mid_blocks(values)
        for index, block in enumerate(self.out_blocks):
            values = block[1](block[0](torch.cat([shortcuts[-index - 1], values], dim=-1)))
        return self.project_out(values[:, 1:])


class _TokensToImage(nn.Module):
    def __init__(self, height: int):
        super().__init__()
        self.height = height

    def forward(self, values: Tensor) -> Tensor:
        batch, tokens, channels = values.shape
        if tokens % self.height:
            raise ValueError("Token count is incompatible with configured state height")
        return values.transpose(1, 2).reshape(
            batch, channels, self.height, tokens // self.height
        )


class _ImageToTokens(nn.Module):
    def forward(self, values: Tensor) -> Tensor:
        return values.flatten(2).transpose(1, 2)


def build_river_vector_field(config: Mapping, *, reference: bool = True):
    """Construct the final wound-specific River vector field from configuration."""
    return RiverVectorFieldRegressor(
        state_size=int(config["state_size"]),
        state_res=tuple(config["state_res"]),
        inner_dim=int(config["inner_dim"]),
        depth=int(config["depth"]),
        mid_depth=int(config["mid_depth"]),
        out_norm=str(config["out_norm"]),
        dropout=float(config["dropout"]),
        reference=reference,
    )


class RiverFlowModel(nn.Module):
    """Final wound River model with an externally supplied VQ-MUSE autoencoder."""

    def __init__(self, config: Mapping, autoencoder: nn.Module):
        super().__init__()
        self.config = config
        model_config = config.get("model", config)
        self.sigma = float(model_config["sigma"])
        self.self_cond_prob = float(model_config["self_cond_prob"])
        self.target_skew_power = float(model_config.get("target_skew_power", 0.5))
        self.ae = autoencoder
        self.vector_field_regressor = build_river_vector_field(
            model_config["vector_field_regressor"]
        )

    def forward(self, observations: Tensor) -> dict[str, Tensor]:
        batch_size, num_observations = observations.shape[:2]
        if num_observations <= 2:
            raise ValueError("River training requires at least three observations")
        uniform = torch.rand(batch_size, device=observations.device)
        target_indices = (
            2 + (num_observations - 2) * uniform.pow(self.target_skew_power)
        ).long().clamp(max=num_observations - 1)
        batch_indices = torch.arange(batch_size, device=observations.device)
        reference_indices = target_indices - 1
        conditioning_indices = torch.stack(
            [torch.randint(0, int(index) - 1, (), device=observations.device) for index in target_indices]
        )
        frames = torch.stack(
            [
                observations[batch_indices, target_indices],
                observations[batch_indices, reference_indices],
                observations[batch_indices, conditioning_indices],
            ],
            dim=1,
        )
        with torch.no_grad():
            self.ae.eval()
            flat = frames.flatten(0, 1)
            encoded = self.ae.encode(flat)
            latents = encoded.reshape(batch_size, 3, *encoded.shape[1:])
        target, reference, conditioning = latents.unbind(dim=1)
        noise = torch.randn_like(target)
        timestamps = torch.rand(
            batch_size, 1, 1, 1, device=target.device, dtype=target.dtype
        )
        inputs = (1 - (1 - self.sigma) * timestamps) * noise + timestamps * target
        targets = (target - (1 - self.sigma) * inputs) / (
            1 - (1 - self.sigma) * timestamps
        )
        distances = (reference_indices - conditioning_indices).to(target.device)
        scalar_times = timestamps.flatten()
        self_conditioning = torch.zeros_like(target)
        if torch.rand((), device=target.device) < self.self_cond_prob:
            with torch.no_grad():
                initial = self.vector_field_regressor(
                    inputs,
                    reference,
                    conditioning,
                    self_conditioning,
                    distances,
                    scalar_times,
                )
                self_conditioning = (
                    (1 - (1 - self.sigma) * timestamps) * initial
                    + (1 - self.sigma) * inputs
                ).detach()
        reconstructed = self.vector_field_regressor(
            inputs,
            reference,
            conditioning,
            self_conditioning,
            distances,
            scalar_times,
        )
        return {
            "observations": observations,
            "reconstructed_vectors": reconstructed,
            "target_vectors": targets,
        }

    @torch.no_grad()
    def generate_frames(
        self,
        observations: Tensor,
        num_frames: int,
        *,
        steps: int = 100,
        warm_start: float = 0.0,
        past_horizon: int = -1,
        verbose: bool = False,
    ) -> Tensor:
        """Autoregressively integrate and decode future VQ-MUSE latents."""
        del verbose
        try:
            from torchdiffeq import odeint
        except ImportError as error:
            raise ImportError("River generation requires torchdiffeq") from error
        flat = observations.flatten(0, 1)
        encoded = self.ae.encode(flat)
        latents = encoded.reshape(*observations.shape[:2], *encoded.shape[1:])
        batch_size, original_count = latents.shape[:2]
        if original_count == 1:
            latents = latents[:, [0, 0]]
        for _ in range(num_frames):
            current_latents = latents

            def vector_field(
                time: Tensor,
                state: Tensor,
                current: Tensor = current_latents,
            ) -> Tensor:
                lower = (
                    0
                    if past_horizon == -1
                    else max(0, len(current[0]) - past_horizon)
                )
                lower = min(lower, len(current[0]) - 2)
                upper = len(current[0]) - 1
                indices = torch.randint(lower, upper, (batch_size,), device=state.device)
                conditioning = current[
                    torch.arange(batch_size, device=state.device), indices
                ]
                return self.vector_field_regressor(
                    state,
                    current[:, -1],
                    conditioning,
                    torch.zeros_like(state),
                    (upper - indices).to(state.device),
                    time * torch.ones(batch_size, device=state.device),
                )
            noise = torch.randn_like(latents[:, -1])
            initial = (1 - (1 - self.sigma) * warm_start) * noise + warm_start * latents[:, -1]
            integration_times = torch.linspace(
                warm_start,
                1,
                max(2, int((1 - warm_start) * steps)),
                device=initial.device,
            )
            predicted = odeint(vector_field, initial, integration_times, method="rk4")[-1]
            latents = torch.cat([latents, self.ae.quantize(predicted).unsqueeze(1)], dim=1)
        if original_count == 1:
            latents = latents[:, 1:]
        flat_latents = latents.flatten(0, 1)
        decoded = self.ae.decode(self.ae.quantize(flat_latents)).clamp(0, 1)
        return decoded.reshape(batch_size, -1, *decoded.shape[1:])


def river_flow_matching_loss(output: Mapping[str, Tensor]) -> Tensor:
    """Final River pointwise conditional-flow-matching objective."""
    return F.mse_loss(output["reconstructed_vectors"], output["target_vectors"])


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
