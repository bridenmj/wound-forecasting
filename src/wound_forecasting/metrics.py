"""Shared image metrics used for the final paper evaluation.

The functions in this module replace several historical notebook versions. They
use one prediction per unique target for the paper comparison and aggregate
PSNR/SSIM over individual prediction-target pairs.
"""

from collections.abc import Mapping

import torch
from torch import Tensor
from torchmetrics.functional.image import structural_similarity_index_measure
from torchmetrics.image.kid import KernelInceptionDistance

ImagePools = Mapping[int, Tensor]


def _as_float_images(images: Tensor) -> Tensor:
    """Return NCHW images as floating point values in ``[0, 1]``."""
    if images.dtype == torch.uint8:
        return images.float() / 255.0
    return images.float().clamp(0, 1)


def _shared_horizons(fake_pools: ImagePools, real_pools: ImagePools) -> list[int]:
    """Return sorted horizons present in both pool mappings."""
    return sorted(set(fake_pools) & set(real_pools))


@torch.inference_mode()
def compute_kid_metrics(
    fake_pools: ImagePools,
    real_pools: ImagePools,
    device: str | torch.device = "cuda",
    subsets: int = 50,
) -> dict:
    """Compute per-horizon and pooled Kernel Inception Distance.

    ``fake_pools[h]`` and ``real_pools[h]`` may contain different numbers of
    images, because KID is a distributional rather than paired metric. The
    ``overall`` result concatenates all eligible horizons, so it is weighted by
    the number of available targets at each horizon. ``overall_unweighted`` is
    retained as a descriptive horizon-balanced summary.

    The standard deviation returned by TorchMetrics describes its internal KID
    subset estimates; it is not a confidence interval over held-out subjects.
    """
    horizons = _shared_horizons(fake_pools, real_pools)
    results: dict = {"per_horizon": {}, "overall": None}

    for horizon in horizons:
        fake = _as_float_images(fake_pools[horizon])
        real = _as_float_images(real_pools[horizon])
        subset_size = min(fake.shape[0], real.shape[0])

        if subset_size < 2:
            results["per_horizon"][horizon] = {
                "mean": float("nan"),
                "std": float("nan"),
                "subset_size": int(subset_size),
            }
            continue

        metric = KernelInceptionDistance(
            subset_size=int(subset_size), subsets=subsets
        ).to(device)
        metric.update((fake * 255).round().to(torch.uint8).to(device), real=False)
        metric.update((real * 255).round().to(torch.uint8).to(device), real=True)
        mean, std = metric.compute()
        results["per_horizon"][horizon] = {
            "mean": float(mean.item()),
            "std": float(std.item()),
            "subset_size": int(subset_size),
        }

    valid_horizons = [
        horizon
        for horizon in horizons
        if not torch.isnan(
            torch.tensor(results["per_horizon"][horizon]["mean"])
        )
    ]

    if valid_horizons:
        results["overall_unweighted"] = {
            "mean": float(
                sum(results["per_horizon"][h]["mean"] for h in valid_horizons)
                / len(valid_horizons)
            ),
            "std": float(
                sum(results["per_horizon"][h]["std"] for h in valid_horizons)
                / len(valid_horizons)
            ),
            "horizons_used": valid_horizons,
        }
    else:
        results["overall_unweighted"] = {
            "mean": float("nan"),
            "std": float("nan"),
            "horizons_used": [],
        }

    eligible = [
        horizon
        for horizon in horizons
        if fake_pools[horizon].shape[0] >= 2
        and real_pools[horizon].shape[0] >= 2
    ]
    if not eligible:
        results["overall"] = {
            "mean": float("nan"),
            "std": float("nan"),
            "subset_size": 0,
        }
        return results

    fake = torch.cat([_as_float_images(fake_pools[h]) for h in eligible])
    real = torch.cat([_as_float_images(real_pools[h]) for h in eligible])
    subset_size = min(fake.shape[0], real.shape[0])
    metric = KernelInceptionDistance(
        subset_size=int(subset_size), subsets=subsets
    ).to(device)
    metric.update((fake * 255).round().to(torch.uint8).to(device), real=False)
    metric.update((real * 255).round().to(torch.uint8).to(device), real=True)
    mean, std = metric.compute()
    results["overall"] = {
        "mean": float(mean.item()),
        "std": float(std.item()),
        "subset_size": int(subset_size),
    }
    return results


@torch.inference_mode()
def compute_targetwise_psnr_ssim(
    fake_pools: ImagePools,
    real_pools: ImagePools,
    device: str | torch.device = "cuda",
) -> dict:
    """Compute PSNR and SSIM for every aligned prediction-target pair.

    The two tensors for each horizon must have identical NCHW shapes. Overall
    values are means over all unique targets, rather than means of horizon-level
    means. This is the aggregation used for the final paper tables.
    """
    horizons = _shared_horizons(fake_pools, real_pools)
    results: dict = {"per_horizon": {}, "overall": {}}
    all_psnr: list[Tensor] = []
    all_ssim: list[Tensor] = []

    for horizon in horizons:
        fake = _as_float_images(fake_pools[horizon])
        real = _as_float_images(real_pools[horizon])
        if fake.shape != real.shape:
            raise ValueError(
                f"Horizon {horizon} is misaligned: fake={tuple(fake.shape)}, "
                f"real={tuple(real.shape)}"
            )

        count = fake.shape[0]
        if count == 0:
            results["per_horizon"][horizon] = {
                "psnr": float("nan"),
                "ssim": float("nan"),
                "n_targets": 0,
            }
            continue

        fake = fake.to(device)
        real = real.to(device)
        mse = (fake - real).square().flatten(start_dim=1).mean(dim=1)
        psnr = 10.0 * torch.log10(1.0 / mse.clamp_min(1e-12))
        ssim = torch.stack(
            [
                structural_similarity_index_measure(
                    fake[index : index + 1],
                    real[index : index + 1],
                    data_range=1.0,
                )
                for index in range(count)
            ]
        ).reshape(-1)

        results["per_horizon"][horizon] = {
            "psnr": float(psnr.mean().item()),
            "ssim": float(ssim.mean().item()),
            "n_targets": int(count),
        }
        all_psnr.append(psnr.cpu())
        all_ssim.append(ssim.cpu())

    if not all_psnr:
        raise ValueError("No aligned prediction-target pairs found")

    psnr = torch.cat(all_psnr)
    ssim = torch.cat(all_ssim)
    results["overall"] = {
        "psnr": float(psnr.mean().item()),
        "ssim": float(ssim.mean().item()),
        "n_targets": int(psnr.numel()),
        "aggregation": "mean_over_unique_targets",
    }
    return results

