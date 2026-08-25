"""Wound-specific token datasets for LLaMA-Adapter training and evaluation."""

from __future__ import annotations

import copy
import json
import random
import re
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset


def day_number(value: str) -> int:
    match = re.search(r"\d+", str(value))
    if match is None:
        raise ValueError(f"Cannot parse day number from {value!r}")
    return int(match.group())


def load_prompts(path: str | Path) -> list[dict[str, Any]]:
    """Load and validate context/target prompt records from JSONL."""
    prompts = []
    with Path(path).open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            prompt = json.loads(line)
            required = {"context", "target", "text_tokens"}
            missing = required - set(prompt)
            if missing:
                raise ValueError(
                    f"Prompt line {line_number} is missing {sorted(missing)}"
                )
            indices = list(prompt["context"]) + list(prompt["target"])
            if not indices or min(indices) < 0:
                raise ValueError(f"Invalid frame indices on line {line_number}")
            prompts.append(prompt)
    if not prompts:
        raise ValueError(f"No prompts found in {path}")
    return prompts


def prepare_text_tokens(
    values,
    *,
    maximum_length: int = 32,
    padding_token: int = 16383,
) -> torch.Tensor:
    """Pad one text-token sequence to the adapter input length."""
    tokens = torch.as_tensor(values, dtype=torch.long).flatten()
    if tokens.numel() > maximum_length:
        tokens = tokens[:maximum_length]
    result = torch.full((maximum_length,), padding_token, dtype=torch.long)
    result[: tokens.numel()] = tokens
    return result


def resolve_image(root: Path, value: str) -> Path:
    path = Path(value)
    if path.is_file():
        return path
    normalized = str(value).replace("\\", "/")
    if "512x512/" in normalized:
        normalized = normalized.split("512x512/", 1)[1]
    candidate = root / normalized.lstrip("/")
    if not candidate.is_file():
        raise FileNotFoundError(f"Image not found: {value!r} -> {candidate}")
    return candidate


def load_rgb_tensor(path: Path, image_size: int = 256) -> torch.Tensor:
    """Load one RGB image using the paper's bilinear resize convention."""
    from PIL import Image
    from torchvision.transforms import InterpolationMode
    from torchvision.transforms.functional import pil_to_tensor, resize

    with Image.open(path) as image:
        image = resize(
            image.convert("RGB"),
            [image_size, image_size],
            interpolation=InterpolationMode.BILINEAR,
        )
        return pil_to_tensor(image).float().div_(255)


@torch.inference_mode()
def encode_manifest_records(
    manifest: dict[str, dict[str, list[str]]],
    *,
    image_root: str | Path,
    vq_model,
    device: str | torch.device,
    image_size: int = 256,
    tokens_per_image: int = 256,
    retain_images: bool = False,
) -> list[dict[str, Any]]:
    """Encode every aligned burst while preserving longitudinal identity."""
    root = Path(image_root)
    records = []
    for wound_id, day_mapping in sorted(manifest.items()):
        days = sorted(day_mapping, key=day_number)
        counts = {len(day_mapping[day]) for day in days}
        if not days or len(counts) != 1:
            raise ValueError(f"Inconsistent burst counts for {wound_id}: {counts}")
        burst_count = counts.pop()
        tokens = {day: [] for day in days}
        images = {day: [] for day in days} if retain_images else None
        ordered = {
            day: sorted(day_mapping[day], key=lambda value: Path(value).name)
            for day in days
        }
        for burst_index in range(burst_count):
            batch = torch.stack(
                [
                    load_rgb_tensor(
                        resolve_image(root, ordered[day][burst_index]), image_size
                    )
                    for day in days
                ]
            )
            encoded = vq_model.encode(batch.to(device))
            if not isinstance(encoded, (tuple, list)) or len(encoded) < 2:
                raise RuntimeError("VQ encoder did not return (..., token_maps)")
            maps = encoded[1].reshape(len(days), -1).detach().cpu().long()
            if maps.shape[1] != tokens_per_image:
                raise ValueError(
                    f"VQ token length is {maps.shape[1]}, expected {tokens_per_image}"
                )
            for day_index, day in enumerate(days):
                tokens[day].append(maps[day_index])
                if images is not None:
                    images[day].append(batch[day_index])
        record = {"wound_id": wound_id, "days": days, "tokens": tokens}
        if images is not None:
            record["images"] = images
        records.append(record)
    return records


class WoundLlamaTrainingDataset(Dataset):
    """Final Pool-A autoregressive dataset used by the paper model."""

    def __init__(
        self,
        records: list[dict[str, Any]],
        prompts: list[dict[str, Any]] | str | Path,
        *,
        number_of_samples: int = 2000,
        maximum_sequence_length: int = 2048,
        maximum_text_length: int = 32,
        tokens_per_image: int = 256,
        image_padding_token: int = 8192,
        text_padding_token: int = 16383,
        ignore_index: int = -100,
    ):
        if not records:
            raise ValueError("Training records are empty")
        self.records = records
        self.prompts = (
            load_prompts(prompts) if isinstance(prompts, (str, Path)) else prompts
        )
        self.number_of_samples = number_of_samples
        self.maximum_sequence_length = maximum_sequence_length
        self.maximum_text_length = maximum_text_length
        self.tokens_per_image = tokens_per_image
        self.image_padding_token = image_padding_token
        self.text_padding_token = text_padding_token
        self.ignore_index = ignore_index
        self.epoch = 0

    def __len__(self) -> int:
        return self.number_of_samples

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def _frame_tokens(self, record, indices, burst_index):
        chunks = []
        for index in indices:
            value = record["tokens"][record["days"][index]][burst_index].reshape(-1)
            if value.numel() != self.tokens_per_image:
                raise ValueError("Unexpected token-map size")
            chunks.append(value.long())
        return torch.cat(chunks)

    def __getitem__(self, index):
        del index
        record = random.choice(self.records)
        sequence_length = len(record["days"])
        eligible = [
            prompt
            for prompt in self.prompts
            if max(prompt["context"] + prompt["target"]) < sequence_length
        ]
        if not eligible:
            raise RuntimeError(
                f"No prompt fits {record['wound_id']} ({sequence_length} frames)"
            )
        prompt = random.choice(eligible)
        first_day = record["days"][0]
        burst_index = random.randrange(len(record["tokens"][first_day]))
        context = self._frame_tokens(record, prompt["context"], burst_index)
        target = self._frame_tokens(record, prompt["target"], burst_index)
        sequence = torch.cat([context, target])
        if sequence.numel() > self.maximum_sequence_length:
            raise ValueError("Prompt exceeds maximum image-token sequence length")
        example = torch.full(
            (self.maximum_sequence_length,), self.image_padding_token, dtype=torch.long
        )
        example[: sequence.numel()] = sequence
        labels = copy.deepcopy(example)
        labels[: context.numel()] = self.ignore_index
        labels[sequence.numel() :] = self.ignore_index
        mask = torch.zeros(self.maximum_sequence_length, dtype=torch.float32)
        mask[: sequence.numel()] = 1
        text = prepare_text_tokens(
            prompt["text_tokens"],
            maximum_length=self.maximum_text_length,
            padding_token=self.text_padding_token,
        )
        return example, labels, mask, text


class WoundLlamaEvaluationDataset(Dataset):
    """Enumerate aligned burst trajectories for fixed four-frame context."""

    def __init__(
        self,
        records: list[dict[str, Any]],
        prompts: list[dict[str, Any]] | str | Path,
        *,
        context_frames: int = 4,
        maximum_text_length: int = 32,
    ):
        self.records = records
        self.prompts = (
            load_prompts(prompts) if isinstance(prompts, (str, Path)) else prompts
        )
        self.context_frames = context_frames
        self.maximum_text_length = maximum_text_length
        self.samples = []
        for record_index, record in enumerate(records):
            bursts = len(record["tokens"][record["days"][0]])
            self.samples.extend((record_index, burst) for burst in range(bursts))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        record_index, burst_index = self.samples[index]
        record = self.records[record_index]
        context = list(range(self.context_frames))
        targets = list(range(self.context_frames, len(record["days"])))
        prompt = next(
            (
                item
                for item in self.prompts
                if item["context"] == context and item["target"] == targets
            ),
            None,
        )
        if prompt is None:
            raise RuntimeError(f"No exact evaluation prompt for {context} -> {targets}")
        maps = torch.stack(
            [record["tokens"][day][burst_index] for day in record["days"]]
        )
        images = None
        if "images" in record:
            images = torch.stack(
                [record["images"][day][burst_index] for day in record["days"]]
            )
        return {
            "wound_id": record["wound_id"],
            "burst_index": burst_index,
            "token_maps": maps,
            "real_images": images,
            "context": context,
            "targets": targets,
            "text_tokens": prepare_text_tokens(
                prompt["text_tokens"], maximum_length=self.maximum_text_length
            ),
        }
