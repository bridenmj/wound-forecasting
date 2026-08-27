#!/usr/bin/env python3
"""Evaluate the released wound LLaMA adapter on held-out trajectories."""

import argparse
import json
from pathlib import Path

import torch

from wound_forecasting.artifacts import resolve_artifact
from wound_forecasting.configuration import load_yaml_config
from wound_forecasting.llama_adapter import load_adapter_delta
from wound_forecasting.llama_data import (
    WoundLlamaEvaluationDataset,
    encode_manifest_records,
)
from wound_forecasting.llama_inference import generate_evaluation_pools
from wound_forecasting.lvm_upstream import construct_wound_llama, load_lvm_components
from wound_forecasting.manifests import DEFAULT_TEST_PIGS, select_pigs
from wound_forecasting.metrics import compute_kid_metrics, compute_targetwise_psnr_ssim


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/llama_adapter_final.yaml")
    parser.add_argument("--lvm-source", required=True)
    parser.add_argument("--vq-source")
    parser.add_argument("--llama-checkpoint-dir", required=True)
    parser.add_argument("--vq-checkpoint-dir", required=True)
    parser.add_argument("--adapter", help="Local adapter-delta checkpoint")
    parser.add_argument("--adapter-repo", default="bridenmj/wound-llama-adapter")
    parser.add_argument("--adapter-file", default="wound_llama_adapter_delta_v1.pth")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--image-root", required=True)
    parser.add_argument("--prompts", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    config = load_yaml_config(
        args.config,
        replacements={
            "IMAGE_ROOT": str(Path(args.image_root).resolve()),
            "MANIFEST_PATH": str(Path(args.manifest).resolve()),
            "PROMPT_PATH": str(Path(args.prompts).resolve()),
        },
    )
    device = torch.device(config["device"])
    adapter_class, get_vq = load_lvm_components(
        args.lvm_source,
        args.vq_source,
    )
    vq_model = get_vq(args.vq_checkpoint_dir).to(device).eval()
    model_cfg = config["model"]
    model = construct_wound_llama(
        adapter_class,
        llama_checkpoint_dir=args.llama_checkpoint_dir,
        maximum_sequence_length=int(model_cfg["max_sequence_length"]),
        maximum_batch_size=int(model_cfg["maximum_batch_size"]),
        image_adapter_length=int(model_cfg["image_adapter_length"]),
        text_adapter_length=int(model_cfg["text_adapter_length"]),
        query_layer=int(model_cfg["query_layer"]),
        text_in_layers=int(model_cfg["text_in_layers"]),
        use_text=bool(model_cfg["use_text"]),
        phase=model_cfg["phase"],
    )
    adapter_path = resolve_artifact(
        args.adapter,
        repository=args.adapter_repo,
        filename=args.adapter_file,
    )
    load_adapter_delta(
        model,
        adapter_path,
        expected_base_model=model_cfg["base_repository"],
    )
    model = model.to(device).eval()

    with Path(args.manifest).open(encoding="utf-8") as stream:
        full_manifest = json.load(stream)
    held_out = select_pigs(
        full_manifest,
        config["data"].get("test_pigs", DEFAULT_TEST_PIGS),
        minimum_days=int(config["data"]["minimum_days"]),
    )
    records = encode_manifest_records(
        held_out,
        image_root=args.image_root,
        vq_model=vq_model,
        device=device,
        retain_images=True,
    )
    dataset = WoundLlamaEvaluationDataset(records, args.prompts)
    generation = config["generation"]
    fake, real = generate_evaluation_pools(
        model,
        vq_model,
        dataset,
        device=device,
        seed=int(config["seed"]),
        temperature=float(generation["temperature"]),
        top_p=float(generation["top_p"]),
    )
    results = {
        "counts": {str(h): int(fake[h].shape[0]) for h in sorted(fake)},
        "kid": compute_kid_metrics(
            fake,
            real,
            device=device,
            subsets=int(generation["kid_subsets"]),
        ),
        "psnr_ssim": compute_targetwise_psnr_ssim(fake, real, device=device),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(results, indent=2))
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()
