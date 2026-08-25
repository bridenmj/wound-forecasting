"""Dataset utilities for variable-context DyneODE latent trajectories."""

from __future__ import annotations

import random
import re
from pathlib import Path

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset


class E4EDirectoryLatentDataset(Dataset):
    """Load longitudinal e4e ``W`` latents from a directory hierarchy.

    Paths must contain ``Day_<n>``, ``ID<n>``, and ``Wound_<id>`` components.
    Files from each day are naturally sorted so that a burst index refers to
    the same within-day position throughout a trajectory.

    Training uses one randomly selected burst per wound. Evaluation sets
    ``enumerate_bursts=True`` to expose every wound/burst trajectory exactly
    once.
    """

    day_pattern = re.compile(r"Day_(\d+)", re.IGNORECASE)
    pig_pattern = re.compile(r"ID\d+", re.IGNORECASE)
    wound_pattern = re.compile(r"Wound_([A-Za-z0-9]+)", re.IGNORECASE)

    def __init__(
        self,
        root_dir: str | Path,
        *,
        length: int | None = None,
        enumerate_bursts: bool = False,
        min_frames: int = 2,
    ) -> None:
        self.root_dir = Path(root_dir).expanduser().resolve()
        self.length = length
        self.enumerate_bursts = enumerate_bursts
        self.min_frames = min_frames
        if not self.root_dir.is_dir():
            raise FileNotFoundError(f"Latent directory not found: {self.root_dir}")
        if length is not None and length < 1:
            raise ValueError("length must be positive or None")
        if enumerate_bursts and length is not None:
            raise ValueError("Deterministic burst evaluation requires length=None")

        self.records = self._build_records()
        self.sample_indices: list[tuple[int, int]] = []
        if enumerate_bursts:
            self.sample_indices = [
                (record_index, burst_index)
                for record_index, record in enumerate(self.records)
                for burst_index in range(record["num_bursts"])
            ]

    def __len__(self) -> int:
        return len(self.sample_indices) if self.enumerate_bursts else len(self.records)

    def __getitem__(self, index: int) -> dict:
        if self.enumerate_bursts:
            record_index, burst_index = self.sample_indices[index]
            record = self.records[record_index]
        else:
            record = self.records[index]
            burst_index = random.randrange(record["num_bursts"])

        days = record["days"]
        if self.length is None:
            selected_indices = np.arange(len(days))
        else:
            if len(days) < self.length:
                raise ValueError(
                    f"{record['wound_id']} has {len(days)} frames; "
                    f"{self.length} required"
                )
            selected_indices = np.sort(
                np.random.choice(len(days), self.length, replace=False)
            )

        selected_days = [days[int(i)] for i in selected_indices]
        latent_paths = [record["latents"][day][burst_index] for day in selected_days]
        return {
            "latents": load_latent_sequence(latent_paths),
            "t_steps": torch.tensor(selected_days, dtype=torch.float32),
            "latent_paths": [str(path) for path in latent_paths],
            "days": selected_days,
            "wound_id": record["wound_id"],
            "burst_idx": burst_index,
        }

    @staticmethod
    def _natural_key(path: Path) -> list[str | int]:
        return [
            int(part) if part.isdigit() else part.lower()
            for part in re.split(r"(\d+)", path.name)
        ]

    def _build_records(self) -> list[dict]:
        latent_paths = sorted(self.root_dir.rglob("*.pt"))
        if not latent_paths:
            raise RuntimeError(f"No .pt latents found under {self.root_dir}")

        grouped: dict[str, dict[int, list[Path]]] = {}
        unparsed: list[str] = []
        for path in latent_paths:
            relative = path.relative_to(self.root_dir).as_posix()
            day_match = self.day_pattern.search(relative)
            pig_match = self.pig_pattern.search(relative)
            wound_match = self.wound_pattern.search(relative)
            if not (day_match and pig_match and wound_match):
                unparsed.append(relative)
                continue
            day = int(day_match.group(1))
            pig = pig_match.group(0).upper()
            wound = wound_match.group(1).upper()
            wound_id = f"{pig}_Wound_{wound}"
            grouped.setdefault(wound_id, {}).setdefault(day, []).append(path)

        if unparsed:
            raise ValueError(
                f"Could not parse {len(unparsed)} latent paths; "
                f"examples: {unparsed[:5]}"
            )

        records: list[dict] = []
        for wound_id, day_map in sorted(grouped.items()):
            days = sorted(day_map)
            if len(days) < self.min_frames:
                continue
            for day in days:
                day_map[day] = sorted(day_map[day], key=self._natural_key)
            burst_counts = {day: len(day_map[day]) for day in days}
            if len(set(burst_counts.values())) != 1:
                raise ValueError(
                    f"Inconsistent burst counts for {wound_id}: {burst_counts}"
                )
            records.append(
                {
                    "wound_id": wound_id,
                    "days": days,
                    "latents": day_map,
                    "num_bursts": burst_counts[days[0]],
                }
            )
        if not records:
            raise RuntimeError(
                f"No trajectories with at least {self.min_frames} frames "
                f"found under {self.root_dir}"
            )
        return records


def load_w_latent(path: str | Path) -> Tensor:
    """Load either Single-W ``[512]`` or broadcast-W ``[L,512]`` storage."""
    value = torch.load(path, map_location="cpu", weights_only=True)
    if isinstance(value, dict):
        if "w" not in value:
            raise ValueError(f"Latent dictionary has no 'w' tensor: {path}")
        value = value["w"]
    if not isinstance(value, Tensor):
        raise TypeError(f"Latent is not a tensor: {path}")
    latent = value.float().squeeze()
    if latent.ndim == 2:
        expected = latent[:1].expand_as(latent)
        if not torch.allclose(latent, expected, atol=1e-5, rtol=1e-5):
            raise ValueError(f"Broadcast-W rows differ in {path}")
        latent = latent[0]
    if latent.shape != (512,):
        raise ValueError(f"Expected [512] in {path}, got {tuple(latent.shape)}")
    if not torch.isfinite(latent).all():
        raise ValueError(f"Non-finite latent values in {path}")
    return latent


def load_latent_sequence(paths: list[str | Path]) -> Tensor:
    """Load an ordered latent trajectory as ``[T,512]``."""
    if not paths:
        raise ValueError("A latent sequence may not be empty")
    return torch.stack([load_w_latent(path) for path in paths])
