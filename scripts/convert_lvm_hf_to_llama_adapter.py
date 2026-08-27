#!/usr/bin/env python3
"""Convert an LVM Hugging Face checkpoint to legacy LLaMA-Adapter format.

Adapted from:
https://github.com/tloen/alpaca-lora/blob/main/export_state_dict_checkpoint.py

The query and key projection weights are unpermuted to match the legacy LLaMA
implementation used by OpenGVLab/LLaMA-Adapter.
"""

import argparse
import json
from pathlib import Path

import torch
from transformers import LlamaForCausalLM

PARAMS_BY_MODEL = {
    "7b": {"dim": 4096, "multiple_of": 256, "n_heads": 32, "n_layers": 32, "norm_eps": 1e-6, "vocab_size": -1},
    "13b": {"dim": 5120, "multiple_of": 256, "n_heads": 40, "n_layers": 40, "norm_eps": 1e-6, "vocab_size": -1},
    "30b": {"dim": 6656, "multiple_of": 256, "n_heads": 52, "n_layers": 60, "norm_eps": 1e-6, "vocab_size": -1},
    "65b": {"dim": 8192, "multiple_of": 256, "n_heads": 64, "n_layers": 80, "norm_eps": 1e-6, "vocab_size": -1},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base-model",
        default="Emma02/LVM_ckpts",
        help="Hugging Face model ID or local Hugging Face checkpoint directory",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory that will receive consolidated.00.pth and params.json",
    )
    parser.add_argument(
        "--size-key",
        choices=sorted(PARAMS_BY_MODEL),
        default="7b",
        help="Legacy LLaMA architecture size",
    )
    return parser.parse_args()


def unpermute(weight: torch.Tensor, *, n_heads: int, dim: int) -> torch.Tensor:
    return (
        weight.view(n_heads, 2, dim // n_heads // 2, dim)
        .transpose(1, 2)
        .reshape(dim, dim)
    )


def translate_state_dict_key(key: str) -> str | None:
    key = key.replace("base_model.model.", "")
    if key == "model.embed_tokens.weight":
        return "tok_embeddings.weight"
    if key == "model.norm.weight":
        return "norm.weight"
    if key == "lm_head.weight":
        return "output.weight"
    if key.startswith("model.layers."):
        layer = key.split(".")[2]
        if key.endswith(".self_attn.q_proj.weight"):
            return f"layers.{layer}.attention.wq.weight"
        if key.endswith(".self_attn.k_proj.weight"):
            return f"layers.{layer}.attention.wk.weight"
        if key.endswith(".self_attn.v_proj.weight"):
            return f"layers.{layer}.attention.wv.weight"
        if key.endswith(".self_attn.o_proj.weight"):
            return f"layers.{layer}.attention.wo.weight"
        if key.endswith(".mlp.gate_proj.weight"):
            return f"layers.{layer}.feed_forward.w1.weight"
        if key.endswith(".mlp.down_proj.weight"):
            return f"layers.{layer}.feed_forward.w2.weight"
        if key.endswith(".mlp.up_proj.weight"):
            return f"layers.{layer}.feed_forward.w3.weight"
        if key.endswith(".input_layernorm.weight"):
            return f"layers.{layer}.attention_norm.weight"
        if key.endswith(".post_attention_layernorm.weight"):
            return f"layers.{layer}.ffn_norm.weight"
        if key.endswith("rotary_emb.inv_freq") or "lora" in key:
            return None
    raise NotImplementedError(f"Unsupported checkpoint key: {key}")


def main() -> None:
    args = parse_args()
    params = PARAMS_BY_MODEL[args.size_key]
    dim = params["dim"]
    n_heads = params["n_heads"]

    model = LlamaForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.float16,
        device_map={"": "cpu"},
    )
    model.eval()

    converted_state = {}
    for key, value in model.state_dict().items():
        converted_key = translate_state_dict_key(key)
        if converted_key is None:
            continue
        if converted_key.endswith(("attention.wq.weight", "attention.wk.weight")):
            value = unpermute(value, n_heads=n_heads, dim=dim)
        converted_state[converted_key] = value.detach().cpu().contiguous()

    expected_tensor_count = 9 * params["n_layers"] + 3
    if len(converted_state) != expected_tensor_count:
        raise RuntimeError(
            f"Expected {expected_tensor_count} converted tensors, "
            f"found {len(converted_state)}"
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = args.output_dir / "consolidated.00.pth"
    params_path = args.output_dir / "params.json"
    torch.save(converted_state, checkpoint_path)
    params_path.write_text(json.dumps(params, indent=2) + "\n", encoding="utf-8")

    print(f"Converted tensors: {len(converted_state)}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Parameters: {params_path}")


if __name__ == "__main__":
    main()
