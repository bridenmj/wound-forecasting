#!/usr/bin/env python3
"""Fast, data-free verification of the cleaned public package."""

import argparse
import platform

import torch

from wound_forecasting.dyneode import ContextConditionedODEFunc, collapse_broadcast_w
from wound_forecasting.manifests import select_pigs
from wound_forecasting.metrics import compute_targetwise_psnr_ssim


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    print("Python:", platform.python_version())
    print("Torch:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())
    print("Selected device:", args.device)

    latent = torch.randn(3, 512)
    broadcast = latent[:, None, :].expand(3, 14, 512).clone()
    assert torch.equal(collapse_broadcast_w(broadcast), latent)

    field = ContextConditionedODEFunc(
        dim=8, hidden_dim=16, depth=3, context_hidden_dim=4
    ).to(args.device)
    context = torch.randn(4, 8, device=args.device)
    times = torch.tensor([0.0, 0.1, 0.4, 1.0], device=args.device)
    encoded = field.encode_context(context, times)
    derivative = field(torch.tensor(1.0, device=args.device), context[-1], encoded)
    assert encoded.shape == derivative.shape == (8,)

    manifest = {
        "ID1325_Wound_I": {f"Day_{i}": [f"512x512/a/{i}.JPG"] for i in range(5)},
        "ID1323_Wound_I": {f"Day_{i}": [f"512x512/b/{i}.JPG"] for i in range(5)},
    }
    assert list(select_pigs(manifest, ["ID1325"])) == ["ID1325_Wound_I"]

    real = {1: torch.zeros(2, 3, 16, 16)}
    fake = {1: torch.full((2, 3, 16, 16), 0.1)}
    metrics = compute_targetwise_psnr_ssim(fake, real, device=args.device)
    assert metrics["overall"]["n_targets"] == 2

    print("DyneODE construction: OK")
    print("Manifest split: OK")
    print("Targetwise metrics: OK")
    print("Environment verification: PASS")


if __name__ == "__main__":
    main()

