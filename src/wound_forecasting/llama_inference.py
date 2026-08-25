"""Aligned LLaMA-Adapter prediction pools for paper evaluation."""

from collections import defaultdict

import torch

from .llama_adapter import decode_token_suffix, generate_token_suffix, seed_generation


@torch.inference_mode()
def generate_evaluation_pools(
    model,
    vq_model,
    dataset,
    *,
    device: str | torch.device,
    seed: int,
    temperature: float = 1.0,
    top_p: float = 0.95,
    tokens_per_image: int = 256,
):
    """Generate one prediction per unique target, keyed by relative horizon."""
    fake, real = defaultdict(list), defaultdict(list)
    model.eval()
    vq_model.eval()
    for sample_index in range(len(dataset)):
        sample = dataset[sample_index]
        token_maps = sample["token_maps"]
        context_count = len(sample["context"])
        image_tokens = token_maps[:context_count].reshape(1, -1).to(device)
        text_tokens = sample["text_tokens"].unsqueeze(0).to(device)
        target_count = len(sample["targets"])
        seed_generation(seed + sample_index)
        suffix = generate_token_suffix(
            model,
            image_tokens,
            text_tokens,
            generated_frames=target_count,
            tokens_per_frame=tokens_per_image,
            temperature=temperature,
            top_p=top_p,
        )
        predictions = decode_token_suffix(vq_model, suffix, tokens_per_image).cpu()
        targets = sample["real_images"]
        if targets is None:
            raise ValueError("Evaluation records must retain real images")
        for offset, target_index in enumerate(sample["targets"]):
            horizon = offset + 1
            fake[horizon].append(predictions[offset])
            real[horizon].append(targets[target_index])
    return (
        {key: torch.stack(value) for key, value in fake.items()},
        {key: torch.stack(value) for key, value in real.items()},
    )
