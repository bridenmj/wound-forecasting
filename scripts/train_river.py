#!/usr/bin/env python3
"""Train the public wound-specific River CFM architecture."""

import argparse
import json
import random
from collections import Counter
from itertools import islice
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

from wound_forecasting.configuration import load_yaml_config
from wound_forecasting.river import RiverFlowModel
from wound_forecasting.river_data import RiverHDF5Sequences
from wound_forecasting.river_training import (
    river_checkpoint_package,
    train_river_step,
    validate_river_loss,
)
from wound_forecasting.vqmuse_upstream import load_vqmuse_autoencoder


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vq-source", required=True)
    parser.add_argument("--config", default="configs/river_final.yaml")
    parser.add_argument("--train-data", required=True)
    parser.add_argument("--validation-data", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--random-seed", type=int, default=1543)
    parser.add_argument("--steps", type=int)
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()
    config = load_yaml_config(args.config)
    random.seed(args.random_seed)
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.random_seed)
    device = torch.device(args.device)
    data_config = config["data"]
    train_data = RiverHDF5Sequences(
        args.train_data,
        image_size=int(data_config["input_size"]),
        minimum_frames=5,
        random_horizontal_flip=bool(data_config["random_horizontal_flip"]),
    )
    validation_data = RiverHDF5Sequences(
        args.validation_data,
        image_size=int(data_config["input_size"]),
        minimum_frames=5,
    )
    counts = Counter(train_data.get_seq_lengths())
    weights = [(max(length - 2, 1)) / counts[length] for length in train_data.lengths]
    training = config["training"]
    optimizer_config = training["optimizer"]
    accumulation = int(optimizer_config["gradient_accumulation_steps"])
    requested_steps = int(args.steps or optimizer_config["num_training_steps"])
    steps = 1 if args.smoke_test else requested_steps
    sampler = WeightedRandomSampler(
        weights,
        num_samples=steps * accumulation,
        replacement=True,
        generator=torch.Generator().manual_seed(args.random_seed),
    )
    loader = DataLoader(train_data, batch_size=1, sampler=sampler, num_workers=0)
    validation_loader = DataLoader(validation_data, batch_size=1, num_workers=0)
    model = RiverFlowModel(config, load_vqmuse_autoencoder(args.vq_source)).to(device)
    optimizer = torch.optim.AdamW(
        model.vector_field_regressor.parameters(),
        lr=float(optimizer_config["learning_rate"]),
        weight_decay=float(optimizer_config["weight_decay"]),
    )
    output = Path(args.output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    model.train()
    optimizer.zero_grad(set_to_none=True)
    global_step = 0
    for microstep, observations in enumerate(loader, start=1):
        loss = train_river_step(
            model,
            observations.to(device),
            optimizer,
            gradient_accumulation_steps=accumulation,
        )
        if microstep % accumulation:
            continue
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        global_step += 1
        print(f"step={global_step} loss={loss:.6f}")
    validation_batches = (
        islice(validation_loader, 1) if args.smoke_test else validation_loader
    )
    validation_loss = validate_river_loss(model, validation_batches, device=device)
    package = river_checkpoint_package(
        model,
        step=global_step,
        config=config,
        validation_loss=validation_loss,
    )
    checkpoint = output / "wound_river_delta_v1.pth"
    torch.save(package, checkpoint)
    (output / "resolved_config.json").write_text(
        json.dumps(config, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Validation loss: {validation_loss:.6f}")
    print(f"Saved: {checkpoint}")


if __name__ == "__main__":
    main()
