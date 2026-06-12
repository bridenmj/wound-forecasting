import os
import random

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

class DyneODELatentDataset(Dataset):
    def __init__(self, full_seq_manifest, base_dir, length=5):
        """
        root_dir: path to latent .pt files
        image_root: root path to RGB images
        " Update
        """
        self.records = self.build_records_from_full_manifest(full_seq_manifest, base_dir)
        self.length = length
        # set in getittemm


    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        record = self.records[idx]
        days = record["days"]
        T_total = len(days)

        if self.length is not None and T_total < self.length:
            raise ValueError(f"Too few frames ({T_total}) for wound {record['wound_id']}")

        day0 = days[0]
        num_bursts = len(record["latents"][day0])
        burst_idx = random.randrange(num_bursts)

        t_idx = np.arange(T_total) if self.length is None else np.sort(np.random.choice(T_total, self.length, replace=False))

        selected_days = [days[i] for i in t_idx]
        latent_paths = [record["latents"][day][burst_idx] for day in selected_days]
        image_paths = [record["images"][day][burst_idx] for day in selected_days]

        latents = self.load_latents(latent_paths)   # [T, 1, 18, 512]

        # use actual day numbers, not just indices
        t_steps = torch.tensor(
            [int(day.split("_")[1]) for day in selected_days],
            dtype=torch.float32
        )

        return {
            "latents": latents.squeeze(1),   # [T, 18, 512]
            "t_steps": t_steps, # Consider using direct indices
            "image_paths": image_paths,
            "days": selected_days,
            "wound_id": record["wound_id"],
            "burst_idx": burst_idx,
        }


    def build_records_from_full_manifest(self, full_seq_manifest, base_dir):
        records = []

        for wound_id, day_dict in full_seq_manifest.items():
            sorted_days = sorted(day_dict.keys(), key=lambda d: int(d.split('_')[1]))
            image_lists = [day_dict[day] for day in sorted_days]   # len=T, each has 5 image paths
            combined = list(zip(*image_lists))                     # len=5 bursts, each burst is length T

            if len(combined) == 0:
                continue

            latents_per_day = {day: [] for day in sorted_days}
            images_per_day = {day: [] for day in sorted_days}

            for burst_idx in range(len(combined)):
                seq_paths = list(combined[burst_idx])   # one burst-consistent trajectory across all days

                latent_seq = [
                    os.path.join(base_dir, os.path.splitext(p)[0] + ".pt")
                    for p in seq_paths
                ]

                for t, day_str in enumerate(sorted_days):
                    latents_per_day[day_str].append(latent_seq[t])
                    images_per_day[day_str].append(seq_paths[t])

            records.append({
                "wound_id": wound_id,
                "days": sorted_days,
                "latents": latents_per_day,   # dict[day] -> list of latent paths, one per burst
                "images": images_per_day,     # optional, if you want aligned RGBs later
            })

        return records

    def load_latents(self, latent_paths):
        # Load and stack all latents
        latents = []
        for path in latent_paths:
            latent = torch.load(path)
            if latent.ndim == 2:
                latent = latent.unsqueeze(0)  # [1, 18, 512]
            latents.append(latent)
        stacked = torch.stack(latents, dim=0)  # → [T, 1, 18, 512]

        return stacked

class DyneODEEvalDataset(Dataset):
    def __init__(
        self,
        full_seq_manifest,
        latent_base_dir,
        image_base_dir,
        image_size=256,
    ):
        """
        full_seq_manifest: manifest from create_datasets_FULL(...), e.g. val_manifest
        latent_base_dir: root containing latent .pt files
        image_base_dir: root containing RGB .JPG files
        image_size: resize GT images to this square size
        """
        self.latent_base_dir = latent_base_dir
        self.image_base_dir = image_base_dir

        self.records = self.build_records_from_full_manifest(
            full_seq_manifest,
            latent_base_dir,
            image_base_dir,
        )

        self.to_tensor = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),   # [0,1], CHW
        ])

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        record = self.records[idx]
        latent_paths = record["latent_paths"]
        image_paths = record["image_paths"]
        days = record["days"]
        T_total = len(days)

        latents = self.load_latents(latent_paths)   # [T, 1, 18, 512] or [T, 1, 14, 512]
        gt_images = self.load_images(image_paths)   # [T, 3, H, W], float in [0,1]

        t_steps = torch.tensor(
            [int(day.split("_")[1]) for day in days],
            dtype=torch.float32
        )

        return {
            "latents": latents.squeeze(1),   # [T, 18, 512] or [T, 14, 512]
            "t_steps": t_steps,              # [T]
            "images": gt_images,             # [T, 3, H, W]
            "seq_len": T_total,
            "image_paths": image_paths,
            "days": days,
            "wound_id": record["wound_id"],
            "burst_idx": record["burst_idx"],
        }

    def build_records_from_full_manifest(self, full_seq_manifest, latent_base_dir, image_base_dir):
        records = []

        for wound_id, day_dict in full_seq_manifest.items():
            sorted_days = sorted(day_dict.keys(), key=lambda d: int(d.split('_')[1]))
            image_lists = [day_dict[day] for day in sorted_days]   # len=T, each has burst-aligned image paths
            combined = list(zip(*image_lists))                     # len=num_bursts, each is a T-long burst trajectory

            if len(combined) == 0:
                continue

            for burst_idx, burst_seq in enumerate(combined):
                seq_paths = list(burst_seq)   # relative JPG paths from manifest

                latent_paths = [
                    os.path.join(latent_base_dir, os.path.splitext(p)[0] + ".pt")
                    for p in seq_paths
                ]

                image_paths = [
                    os.path.join(image_base_dir, p)
                    for p in seq_paths
                ]

                records.append({
                    "wound_id": wound_id,
                    "days": sorted_days,
                    "burst_idx": burst_idx,
                    "latent_paths": latent_paths,
                    "image_paths": image_paths,
                })

        return records

    def load_latents(self, latent_paths):
        latents = []
        for path in latent_paths:
            latent = torch.load(path)
            if latent.ndim == 2:
                latent = latent.unsqueeze(0)  # [1, 18, 512] or [1, 14, 512]
            latents.append(latent)
        return torch.stack(latents, dim=0)     # [T, 1, 18, 512]

    def load_images(self, image_paths):
        imgs = []
        for path in image_paths:
            img = Image.open(path).convert("RGB")
            img = self.to_tensor(img)          # [3, H, W], float in [0,1]
            imgs.append(img)
        return torch.stack(imgs, dim=0)        # [T, 3, H, W]

