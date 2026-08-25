#!/usr/bin/env python3
"""Evaluate the released River model on model-ready HDF5 trajectories."""

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from wound_forecasting.artifacts import resolve_artifact
from wound_forecasting.metrics import compute_kid_metrics, compute_targetwise_psnr_ssim
from wound_forecasting.river import generate_river_pools, load_river_weights


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--river-source", required=True)
    parser.add_argument("--shared-source", required=True)
    parser.add_argument("--config", default="configs/river_final.yaml")
    parser.add_argument("--data", required=True)
    parser.add_argument("--checkpoint")
    parser.add_argument("--checkpoint-repo", default="bridenmj/wound-river")
    parser.add_argument(
        "--checkpoint-file",
        help="Exact filename in --checkpoint-repo when --checkpoint is omitted",
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    river = Path(args.river_source).expanduser().resolve()
    shared = Path(args.shared_source).expanduser().resolve()
    sys.path[:0] = [str(river), str(shared)]
    from dataset.video_dataset import FullSequenceEvalDataset, pad_collate_fullseq
    from lutils.configuration import Configuration
    from model import Model

    configuration = Configuration(str(Path(args.config).resolve()))
    model = Model(configuration["model"])
    if args.checkpoint is None and args.checkpoint_file is None:
        parser.error("provide --checkpoint or --checkpoint-file")
    checkpoint = resolve_artifact(
        args.checkpoint,
        repository=args.checkpoint_repo,
        filename=args.checkpoint_file or "",
    )
    load_river_weights(model, checkpoint)
    device = torch.device(args.device)
    model = model.to(device).eval()
    data_config = configuration["data"]
    dataset = FullSequenceEvalDataset(
        data_path=args.data,
        input_size=data_config["input_size"],
        crop_size=data_config["crop_size"],
        skip_frames=data_config["skip_frames"],
        use_albumentations=False,
    )
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        collate_fn=pad_collate_fullseq,
    )
    evaluation = configuration["evaluation"]
    fake, real = generate_river_pools(
        model,
        loader,
        device=device,
        steps=int(evaluation["steps"]),
        context_frames=int(evaluation["condition_frames"]),
        past_horizon=int(evaluation.get("past_horizon", -1)),
    )
    results = {
        "counts": {str(h): int(fake[h].shape[0]) for h in sorted(fake)},
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
