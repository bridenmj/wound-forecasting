"""HDF5 sequence datasets used by the wound-specific River experiments."""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import Dataset


class RiverHDF5Sequences(Dataset):
    """Read complete variable-length image trajectories from released HDF5 data."""

    def __init__(
        self,
        path: str | Path,
        *,
        image_size: int = 256,
        minimum_frames: int = 1,
        random_horizontal_flip: bool = False,
    ):
        try:
            import h5py
        except ImportError as error:
            raise ImportError("River HDF5 data requires h5py") from error
        self.h5py = h5py
        self.path = Path(path).expanduser().resolve()
        if not self.path.is_file():
            raise FileNotFoundError(self.path)
        self.image_size = image_size
        self.minimum_frames = minimum_frames
        self.random_horizontal_flip = random_horizontal_flip
        with h5py.File(self.path, "r") as handle:
            self.indices = sorted(handle["len"], key=int)
            self.lengths = [int(handle["len"][index][()]) for index in self.indices]
        if any(length < minimum_frames for length in self.lengths):
            raise ValueError("HDF5 file contains a trajectory shorter than minimum_frames")

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> Tensor:
        key = self.indices[index]
        with self.h5py.File(self.path, "r") as handle:
            frames = [
                torch.from_numpy(handle[key][str(frame)][()])
                for frame in range(self.lengths[index])
            ]
        values = torch.stack(frames).permute(0, 3, 1, 2).float() / 255.0
        if values.shape[-2:] != (self.image_size, self.image_size):
            values = F.interpolate(
                values,
                size=(self.image_size, self.image_size),
                mode="bilinear",
                align_corners=False,
                antialias=True,
            )
        if self.random_horizontal_flip and torch.rand(()) < 0.5:
            values = values.flip(-1)
        return values

    def get_seq_lengths(self) -> list[int]:
        return list(self.lengths)


def pad_river_sequences(batch: list[Tensor]) -> tuple[Tensor, Tensor]:
    """Pad variable-length River sequences and return their true lengths."""
    if not batch:
        raise ValueError("Cannot collate an empty batch")
    lengths = torch.tensor([len(sequence) for sequence in batch], dtype=torch.long)
    maximum = int(lengths.max())
    padded = batch[0].new_zeros((len(batch), maximum, *batch[0].shape[1:]))
    for index, sequence in enumerate(batch):
        padded[index, : len(sequence)] = sequence
    return padded, lengths
