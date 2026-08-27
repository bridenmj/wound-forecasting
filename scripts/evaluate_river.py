#!/usr/bin/env python3
"""Evaluate the public wound River architecture on released HDF5 trajectories."""

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from wound_forecasting.artifacts import resolve_artifact
from wound_forecasting.configuration import load_yaml_config
from wound_forecasting.metrics import compute_kid_metrics, compute_targetwise_psnr_ssim
from wound_forecasting.river import (
    RiverFlowModel,
    generate_river_pools,
    load_river_weights,
)
from wound_forecasting.river_data import RiverHDF5Sequences, pad_river_sequences
from wound_forecasting.vqmuse_upstream import load_vqmuse_autoencoder


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vq-source", required=True)
    parser.add_argument("--config", default="configs/river_final.yaml")
    parser.add_argument("--data", required=True)
    parser.add_argument("--checkpoint")
    parser.add_argument("--checkpoint-repo", default="bridenmj/wound-river")
    parser.add_argument("--checkpoint-file", default="wound_river_delta_v1.pth")
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    config = load_yaml_config(args.config)
    device = torch.device(args.device)
    model = RiverFlowModel(config, load_vqmuse_autoencoder(args.vq_source))
    checkpoint = resolve_artifact(
        args.checkpoint,
        repository=args.checkpoint_repo,
        filename=args.checkpoint_file,
    )
    load_river_weights(model, checkpoint)
    model = model.to(device).eval()
    dataset = RiverHDF5Sequences(
        args.data,
        image_size=int(config["data"]["input_size"]),
        minimum_frames=int(config["evaluation"]["condition_frames"]) + 1,
    )
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        collate_fn=pad_river_sequences,
    )
    evaluation = config["evaluation"]
    fake, real = generate_river_pools(
        model,
        loader,
        device=device,
        steps=int(evaluation["steps"]),
        context_frames=int(evaluation["condition_frames"]),
        past_horizon=int(evaluation.get("past_horizon", -1)),
    )
    results = {
        "counts": {str(horizon): int(fake[horizon].shape[0]) for horizon in fake},
        "kid": compute_kid_metrics(fake, real, device=device, subsets=50),
        "psnr_ssim": compute_targetwise_psnr_ssim(fake, real, device=device),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(results, indent=2))
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()
