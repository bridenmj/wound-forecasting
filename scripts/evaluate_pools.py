#!/usr/bin/env python3
"""Compute the common paper metrics from previously generated image pools."""

import argparse
import json
from pathlib import Path

import torch

from wound_forecasting.metrics import compute_kid_metrics, compute_targetwise_psnr_ssim


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("pools", type=Path, help="Torch file with fake_pools and real_pools")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--kid-subsets", type=int, default=50)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = torch.load(args.pools, map_location="cpu", weights_only=True)
    fake = payload["fake_pools"]
    real = payload["real_pools"]
    if set(fake) != set(real):
        raise ValueError("fake_pools and real_pools must contain the same horizons")
    for horizon in fake:
        if fake[horizon].shape != real[horizon].shape:
            raise ValueError(
                f"H{horizon} is not one-to-one: {fake[horizon].shape} vs {real[horizon].shape}"
            )

    result = {
        "kid": compute_kid_metrics(fake, real, args.device, args.kid_subsets),
        "psnr_ssim": compute_targetwise_psnr_ssim(fake, real, args.device),
        "protocol": {
            "unit": "unique_prediction_target_pair",
            "kid_subsets": args.kid_subsets,
            "horizons": sorted(fake),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()

