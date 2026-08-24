"""Project-specific LLaMA-Adapter image-generation helpers.

The upstream LLaMA-Adapter implementation is not duplicated here. These
functions capture the wound-forecasting additions at its public boundary.
"""

import random

import numpy as np
import torch
from torch import Tensor


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
        raise RuntimeError(f"Generated suffix has {tuple(suffix.shape)}; expected {expected}")
    return suffix


@torch.inference_mode()
def decode_token_suffix(vq_model, suffix: Tensor, tokens_per_frame: int = 256) -> Tensor:
    """Decode generated tokens and normalize decoder output to ``[0,1]``."""
    images = vq_model.decode_code(suffix.reshape(-1, tokens_per_frame)).float()
    if images.min().item() < 0:
        return (images.clamp(-1, 1) + 1) / 2
    return images.clamp(0, 1)

