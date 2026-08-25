#!/usr/bin/env python3
"""Train the final variable-context DyneODE experiment."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import ConcatDataset, DataLoader

from wound_forecasting.artifacts import resolve_artifact
from wound_forecasting.configuration import load_yaml_config
from wound_forecasting.dyneode import ContextConditionedODEFunc
from wound_forecasting.dyneode_data import E4EDirectoryLatentDataset
from wound_forecasting.dyneode_training import (
    checkpoint_package,
    train_trajectory_step,
    validate_ode,
)
from wound_forecasting.stylegan import load_wound_stylegan


def mean_rows(rows: list[dict[str, float]]) -> dict[str, float]:
    keys = rows[0]
    return {key: sum(row[key] for row in rows) / len(rows) for key in keys}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/dyneode_final.yaml")
    parser.add_argument("--latent-root", required=True)
    parser.add_argument("--stylegan-source", required=True)
    parser.add_argument("--stylegan-checkpoint")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()

    config = load_yaml_config(
        args.config, replacements={"LATENT_ROOT": str(Path(args.latent_root).resolve())}
    )
    device = torch.device(args.device or config["device"])
    seed = int(config["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    model_config = config["model"]
    training = config["training"]
    data_config = config["data"]
    output = Path(args.output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)

    train_data = E4EDirectoryLatentDataset(
        Path(args.latent_root) / data_config["train_split"], min_frames=2
    )
    development_validation = E4EDirectoryLatentDataset(
        Path(args.latent_root) / data_config["development_validation_split"],
        min_frames=2,
    )
    if data_config["fold_development_validation_into_training"]:
        training_data = ConcatDataset([train_data, development_validation])
        validation_root = Path(args.latent_root) / data_config["held_out_split"]
    else:
        training_data = train_data
        validation_root = (
            Path(args.latent_root) / data_config["development_validation_split"]
        )
    validation_data = E4EDirectoryLatentDataset(
        validation_root, enumerate_bursts=True, min_frames=2
    )
    train_loader = DataLoader(training_data, batch_size=1, shuffle=True, num_workers=0)
    validation_loader = DataLoader(
        validation_data, batch_size=1, shuffle=False, num_workers=0
    )

    odefunc = ContextConditionedODEFunc(
        dim=int(model_config["style_dim"]),
        hidden_dim=int(model_config["hidden_dim"]),
        depth=int(model_config["depth"]),
        context_encoder_type=model_config["context_encoder_type"],
        context_hidden_dim=int(model_config["context_hidden_dim"]),
    ).to(device)
    optimizer = torch.optim.AdamW(
        odefunc.parameters(),
        lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )

    generator_config = config["generator"]
    generator_checkpoint = resolve_artifact(
        args.stylegan_checkpoint,
        repository=generator_config["repository"],
        filename=generator_config["filename"],
    )
    generator, _ = load_wound_stylegan(
        generator_checkpoint,
        source_root=args.stylegan_source,
        device=device,
        image_size=int(generator_config["image_size"]),
        style_dim=int(generator_config["style_dim"]),
        mapping_network_depth=int(generator_config["mapping_network_depth"]),
        channel_multiplier=int(generator_config["channel_multiplier"]),
    )
    try:
        import lpips
        from torchdiffeq import odeint
    except ImportError as error:
        raise RuntimeError("Install the 'dyneode' optional dependencies") from error
    perceptual = lpips.LPIPS(net=training["lpips_network"]).to(device).eval()

    epochs = 1 if args.smoke_test else int(args.epochs or training["epochs"])
    best_value = float("inf")
    best_package = None
    global_step = 0
    patience_count = 0
    rng = random.Random(seed)
    checkpoint_metadata = {
        "model_type": "time_context_w_gru_ode",
        "context_encoder_type": model_config["context_encoder_type"],
        "context_hidden_dim": int(model_config["context_hidden_dim"]),
        "hidden_dim": int(model_config["hidden_dim"]),
        "depth": int(model_config["depth"]),
        "style_dim": int(model_config["style_dim"]),
        "time_scale": float(model_config["time_scale"]),
        "min_context_size": int(model_config["min_context_size"]),
        "max_context_size": int(model_config["max_context_size"]),
        "visualization_context_size": int(model_config["visualization_context_size"]),
        "selection_metric": training["selection_metric"],
    }

    for epoch in range(epochs):
        odefunc.train()
        rows = []
        for batch in train_loader:
            rows.append(
                train_trajectory_step(
                    odefunc=odefunc,
                    optimizer=optimizer,
                    latents=batch["latents"][0].to(device),
                    times=batch["t_steps"][0].to(device),
                    odeint_fn=odeint,
                    generator=generator,
                    perceptual_metric=perceptual,
                    min_context_size=int(model_config["min_context_size"]),
                    max_context_size=int(model_config["max_context_size"]),
                    time_scale=float(model_config["time_scale"]),
                    latent_loss_weight=float(training["latent_loss_weight"]),
                    perceptual_loss_weight=float(training["perceptual_loss_weight"]),
                    latent_loss_scale=float(training["latent_loss_scale"]),
                    perceptual_loss_scale=float(training["perceptual_loss_scale"]),
                    rng=rng,
                )
            )
            global_step += 1
            if args.smoke_test:
                break
        train_metrics = mean_rows(rows)
        print(f"epoch={epoch + 1} train={json.dumps(train_metrics, sort_keys=True)}")

        should_validate = (
            args.smoke_test
            or epoch == 0
            or (epoch + 1) % int(training["validation_every"]) == 0
            or epoch + 1 == epochs
        )
        if not should_validate:
            continue
        validation_metrics = validate_ode(
            odefunc,
            validation_loader,
            device=device,
            odeint_fn=odeint,
            time_scale=float(model_config["time_scale"]),
            min_context_size=int(model_config["min_context_size"]),
            max_context_size=int(model_config["max_context_size"]),
            generator=generator,
            perceptual_metric=perceptual,
            latent_loss_scale=float(training["latent_loss_scale"]),
            perceptual_loss_scale=float(training["perceptual_loss_scale"]),
            latent_loss_weight=float(training["latent_loss_weight"]),
            perceptual_loss_weight=float(training["perceptual_loss_weight"]),
        )
        print(
            f"epoch={epoch + 1} validation={json.dumps(validation_metrics, sort_keys=True)}"
        )
        selection = float(validation_metrics[training["selection_metric"]])
        if selection < best_value:
            best_value = selection
            patience_count = 0
            best_package = checkpoint_package(
                odefunc,
                epoch=epoch + 1,
                global_step=global_step,
                validation_metrics=validation_metrics,
                config=checkpoint_metadata,
            )
            torch.save(best_package, output / "best_val_checkpoint.pth")
        else:
            patience_count += 1
        if patience_count >= int(training["early_stopping_patience"]):
            break

    if best_package is None:
        raise RuntimeError("Training completed without a validation checkpoint")
    (output / "resolved_config.json").write_text(
        json.dumps(config, indent=2) + "\n", encoding="utf-8"
    )
    print("Best checkpoint:", output / "best_val_checkpoint.pth")


if __name__ == "__main__":
    main()
