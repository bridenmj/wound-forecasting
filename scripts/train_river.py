#!/usr/bin/env python3
"""Public entry point for an external River training implementation."""

import argparse
import runpy
import sys
import tempfile
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--river-source",
        required=True,
        help="Path to the River source tree containing train.py",
    )
    parser.add_argument(
        "--shared-source",
        help="Optional parent added to Python's path for Shared dependencies",
    )
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--config", default="configs/river_final.yaml")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--num-gpus", type=int)
    parser.add_argument("--resume-step", type=int)
    parser.add_argument("--random-seed", type=int, default=1543)
    parser.add_argument("--wandb", action="store_true")
    args = parser.parse_args()

    river = Path(args.river_source).expanduser().resolve()
    train_script = river / "train.py"
    if not train_script.is_file():
        raise FileNotFoundError(f"River training entry point not found: {train_script}")

    import_paths = [str(river)]
    if args.shared_source:
        shared_parent = Path(args.shared_source).expanduser().resolve()
        if not shared_parent.is_dir():
            raise FileNotFoundError(
                f"Shared dependency directory not found: {shared_parent}"
            )
        import_paths.append(str(shared_parent))
    sys.path[:0] = import_paths
    config = Path(args.config).read_text(encoding="utf-8").replace(
        "${DATA_ROOT}", str(Path(args.data_root).resolve())
    )
    with tempfile.TemporaryDirectory() as directory:
        resolved_config = Path(directory) / "river_final.yaml"
        resolved_config.write_text(config, encoding="utf-8")
        forwarded = [
            str(train_script),
            "--run-name",
            args.run_name,
            "--config",
            str(resolved_config),
        ]
        if args.num_gpus is not None:
            forwarded += ["--num-gpus", str(args.num_gpus)]
        if args.resume_step is not None:
            forwarded += ["--resume-step", str(args.resume_step)]
        forwarded += ["--random-seed", str(args.random_seed)]
        if args.wandb:
            forwarded.append("--wandb")
        sys.argv = forwarded
        runpy.run_path(str(train_script), run_name="__main__")


if __name__ == "__main__":
    main()
