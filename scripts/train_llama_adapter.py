#!/usr/bin/env python3
"""Train the wound LLaMA adapter using an external LVM source checkout."""

import argparse
import random
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch.utils.data import DataLoader

from wound_forecasting.configuration import load_yaml_config
from wound_forecasting.llama_data import (
    WoundLlamaTrainingDataset,
    encode_manifest_records,
)
from wound_forecasting.llama_training import save_adapter_delta
from wound_forecasting.lvm_upstream import construct_wound_llama, load_lvm_components
from wound_forecasting.manifests import DEFAULT_TRAIN_PIGS, select_pigs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/llama_adapter_final.yaml")
    parser.add_argument("--lvm-source", required=True)
    parser.add_argument("--vq-source")
    parser.add_argument("--llama-checkpoint-dir", required=True)
    parser.add_argument("--vq-checkpoint-dir", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--image-root", required=True)
    parser.add_argument("--prompts", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()

    import json
    import sys

    config = load_yaml_config(
        args.config,
        replacements={
            "IMAGE_ROOT": str(Path(args.image_root).resolve()),
            "MANIFEST_PATH": str(Path(args.manifest).resolve()),
            "PROMPT_PATH": str(Path(args.prompts).resolve()),
        },
    )
    seed = int(config["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    device = torch.device(config["device"])

    adapter_class, get_vq = load_lvm_components(
        args.lvm_source,
        args.vq_source,
    )
    source = str(Path(args.lvm_source).expanduser().resolve())
    if source not in sys.path:
        sys.path.insert(0, source)
    from engine_finetune import train_one_epoch
    from util import misc
    from util.misc import NativeScalerWithGradNormCount

    vq_model = get_vq(args.vq_checkpoint_dir).to(device).eval()
    with Path(args.manifest).open(encoding="utf-8") as stream:
        full_manifest = json.load(stream)
    manifest = select_pigs(
        full_manifest,
        config["data"].get("train_pigs", DEFAULT_TRAIN_PIGS),
        minimum_days=int(config["data"]["minimum_days"]),
    )
    records = encode_manifest_records(
        manifest,
        image_root=args.image_root,
        vq_model=vq_model,
        device=device,
    )
    dataset = WoundLlamaTrainingDataset(
        records,
        args.prompts,
        number_of_samples=(
            min(int(config["data"]["num_samples"]), 4)
            if args.smoke_test
            else int(config["data"]["num_samples"])
        ),
        maximum_sequence_length=int(config["model"]["max_sequence_length"]),
    )
    loader = DataLoader(
        dataset,
        batch_size=int(config["training"]["batch_size"]),
        shuffle=True,
        num_workers=int(config["data"]["workers"]),
        pin_memory=device.type == "cuda",
        drop_last=True,
    )
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
    ).to(device)
    training = config["training"]
    parameter_groups = misc.add_weight_decay(model, float(training["weight_decay"]))
    optimizer = torch.optim.AdamW(
        parameter_groups,
        lr=float(training["learning_rate"]),
        betas=(0.9, 0.95),
    )
    scaler = NativeScalerWithGradNormCount()
    engine_args = SimpleNamespace(
        accum_iter=int(training["gradient_accumulation_steps"]),
        lr=float(training["learning_rate"]),
        min_lr=float(training["minimum_learning_rate"]),
        warmup_epochs=0,
        epochs=1 if args.smoke_test else int(training["epochs"]),
    )
    epochs = engine_args.epochs
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    for epoch in range(epochs):
        dataset.set_epoch(epoch)
        train_one_epoch(
            model,
            loader,
            optimizer,
            device,
            epoch,
            scaler,
            args=engine_args,
        )
        save_adapter_delta(
            output,
            model,
            base_model=model_cfg["base_repository"],
            architecture=model_cfg,
            epoch=epoch,
        )
    print(f"Saved adapter delta: {output}")


if __name__ == "__main__":
    main()
