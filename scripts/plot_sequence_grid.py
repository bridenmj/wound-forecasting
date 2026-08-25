#!/usr/bin/env python3
"""Plot one eight-frame tensor sequence without notebook dependencies."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torchvision.utils import make_grid


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("tensor", type=Path, help="Torch tensor shaped [8,3,H,W]")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    images = torch.load(args.tensor, map_location="cpu", weights_only=True).float().clamp(0, 1)
    if tuple(images.shape[:2]) != (8, 3):
        raise ValueError(f"Expected [8,3,H,W], received {tuple(images.shape)}")
    grid = make_grid(images, nrow=8, padding=0)
    height, width = grid.shape[-2:]
    figure, axis = plt.subplots(figsize=(16, 16 * height / width))
    axis.imshow(grid.permute(1, 2, 0).numpy(), interpolation="nearest")
    axis.axis("off")
    figure.subplots_adjust(left=0, right=1, top=1, bottom=0)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=300, bbox_inches="tight", pad_inches=0)
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()

