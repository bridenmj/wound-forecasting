#!/usr/bin/env python3
"""Evaluate the released DyneODE checkpoint on latent trajectories."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from torch.utils.data import DataLoader

from wound_forecasting.artifacts import resolve_artifact
from wound_forecasting.dyneode_data import E4EDirectoryLatentDataset
from wound_forecasting.dyneode_inference import (
    build_dyneode_image_pools,
    load_dyneode_checkpoint,
)
from wound_forecasting.metrics import compute_kid_metrics, compute_targetwise_psnr_ssim
from wound_forecasting.stylegan import load_wound_stylegan


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--latent-root", required=True)
    parser.add_argument("--stylegan-source", required=True)
    parser.add_argument("--dyneode-checkpoint")
    parser.add_argument("--stylegan-checkpoint")
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--context-size", type=int, default=4)
    args = parser.parse_args()

    dyneode_path = resolve_artifact(
        args.dyneode_checkpoint,
        repository="bridenmj/wound-dyneode",
        filename="best_val_checkpoint.pth",
    )
    stylegan_path = resolve_artifact(
        args.stylegan_checkpoint,
        repository="bridenmj/wound-stylegan",
        filename="network-snapshot-005000_jmir_50000_redo_.pt",
    )
    odefunc, package = load_dyneode_checkpoint(dyneode_path, device=args.device)
    generator, _ = load_wound_stylegan(
        stylegan_path, source_root=args.stylegan_source, device=args.device
    )
    try:
        from torchdiffeq import odeint
    except ImportError as error:
        raise RuntimeError("Install torchdiffeq for DyneODE evaluation") from error
    dataset = E4EDirectoryLatentDataset(
        args.latent_root, enumerate_bursts=True, min_frames=args.context_size + 1
    )
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    fake, real = build_dyneode_image_pools(
        odefunc,
        loader,
        generator=generator,
        odeint_fn=odeint,
        device=args.device,
        context_size=args.context_size,
        time_scale=float(package["time_scale"]),
    )
    result = {
        "kid": compute_kid_metrics(fake, real, device=args.device, subsets=50),
        "psnr_ssim": compute_targetwise_psnr_ssim(fake, real, device=args.device),
        "counts": {str(horizon): len(images) for horizon, images in fake.items()},
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
