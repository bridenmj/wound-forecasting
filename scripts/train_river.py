#!/usr/bin/env python3
"""Public entry point for the captured River training implementation."""

import argparse
import runpy
import sys
import tempfile
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--config", default="configs/river_final.yaml")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--num-gpus", type=int)
    parser.add_argument("--resume-step", type=int)
    parser.add_argument("--random-seed", type=int, default=1543)
    parser.add_argument("--wandb", action="store_true")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    river = root / "paper_snapshot" / "source" / "river"
    shared_parent = root / "paper_snapshot" / "source"
    sys.path[:0] = [str(river), str(shared_parent)]
    config = Path(args.config).read_text(encoding="utf-8").replace(
        "${DATA_ROOT}", str(Path(args.data_root).resolve())
    )
    with tempfile.TemporaryDirectory() as directory:
        resolved_config = Path(directory) / "river_final.yaml"
        resolved_config.write_text(config, encoding="utf-8")
        forwarded = [
            str(river / "train.py"),
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
        runpy.run_path(str(river / "train.py"), run_name="__main__")


if __name__ == "__main__":
    main()
