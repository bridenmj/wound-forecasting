"""Project-specific LLaMA-Adapter image-generation helpers.

The upstream LLaMA-Adapter implementation is not duplicated here. These
functions capture the wound-forecasting additions at its public boundary.
"""

import random
from collections.abc import Mapping
from os import PathLike
from typing import Any

import numpy as np
import torch
from torch import Tensor


ADAPTER_DELTA_FORMAT = "wound_forecasting_adapter_delta_v1"


def load_adapter_delta(
    model,
    checkpoint: str | PathLike[str] | Mapping[str, Any],
    *,
    expected_base_model: str | None = None,
    map_location: str | torch.device = "cpu",
):
    """Overlay a verified wound-forecasting adapter on an upstream LVM model.

    The checkpoint contains only project-trained parameters; missing keys from
    ``load_state_dict`` are therefore expected to be the unchanged base-model
    tensors. Adapter keys, however, are validated strictly and may not be
    unknown, skipped, or changed during loading.

    Returns the checkpoint package and PyTorch load result for provenance and
    logging. Construct ``model`` with the configuration stored in
    ``package["config"]`` before calling this function.
    """
    if isinstance(checkpoint, Mapping):
        package = dict(checkpoint)
    else:
        package = torch.load(
            checkpoint,
            map_location=map_location,
            weights_only=True,
        )

    if not isinstance(package, dict):
        raise TypeError("Adapter checkpoint must contain a dictionary package")
    if package.get("format") != ADAPTER_DELTA_FORMAT:
        raise ValueError(
            f"Unsupported adapter format {package.get('format')!r}; "
            f"expected {ADAPTER_DELTA_FORMAT!r}"
        )

    config = package.get("config")
    if not isinstance(config, Mapping):
        raise TypeError("Adapter checkpoint has no configuration mapping")
    base_model = config.get("base_model")
    if expected_base_model is not None and base_model != expected_base_model:
        raise ValueError(
            f"Adapter expects base model {base_model!r}, not "
            f"{expected_base_model!r}"
        )

    adapter_state = package.get("model")
    if not isinstance(adapter_state, Mapping) or not adapter_state:
        raise ValueError("Adapter checkpoint contains no model tensors")
    non_tensor_keys = [
        key for key, value in adapter_state.items() if not isinstance(value, Tensor)
    ]
    if non_tensor_keys:
        raise TypeError(f"Adapter values are not tensors: {non_tensor_keys}")

    model_state = model.state_dict()
    unknown_keys = sorted(set(adapter_state) - set(model_state))
    if unknown_keys:
        raise KeyError(f"Adapter keys are absent from the model: {unknown_keys}")

    load_result = model.load_state_dict(adapter_state, strict=False)
    if load_result.unexpected_keys:
        raise RuntimeError(
            f"Unexpected adapter keys: {sorted(load_result.unexpected_keys)}"
        )
    unloaded_adapter_keys = sorted(set(adapter_state) & set(load_result.missing_keys))
    if unloaded_adapter_keys:
        raise RuntimeError(f"Adapter tensors were not loaded: {unloaded_adapter_keys}")

    loaded_state = model.state_dict()
    mismatched = [
        key
        for key, saved in adapter_state.items()
        if not torch.equal(loaded_state[key].detach().cpu(), saved.detach().cpu())
    ]
    if mismatched:
        raise RuntimeError(
            f"Loaded adapter tensors differ from checkpoint: {mismatched}"
        )

    return package, load_result


def seed_generation(seed: int) -> None:
    """Seed Python, NumPy, and Torch generation consistently."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@torch.inference_mode()
def generate_token_suffix(
    model,
    image_tokens: Tensor,
    text_tokens: Tensor,
    *,
    generated_frames: int,
    tokens_per_frame: int = 256,
    temperature: float = 1.0,
    top_p: float = 0.95,
) -> Tensor:
    """Generate and return only the future image-token suffix."""
    generated_length = generated_frames * tokens_per_frame
    sequence = model.generate(
        img_tokens=image_tokens,
        text_tokens=text_tokens,
        max_gen_len=generated_length,
        temperature=temperature,
        top_p=top_p,
    )
    start = image_tokens.shape[1]
    suffix = sequence[:, start : start + generated_length]
    expected = (image_tokens.shape[0], generated_length)
    if tuple(suffix.shape) != expected:
        raise RuntimeError(
            f"Generated suffix has {tuple(suffix.shape)}; expected {expected}"
        )
    return suffix


@torch.inference_mode()
def decode_token_suffix(
    vq_model,
    suffix: Tensor,
    tokens_per_frame: int = 256,
) -> Tensor:
    """Decode generated tokens and normalize decoder output to ``[0,1]``."""
    images = vq_model.decode_code(suffix.reshape(-1, tokens_per_frame)).float()
    if images.min().item() < 0:
        return (images.clamp(-1, 1) + 1) / 2
    return images.clamp(0, 1)
