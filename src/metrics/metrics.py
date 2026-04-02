import torch
from torchmetrics.image.kid import KernelInceptionDistance
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
import torch_fidelity

@torch.no_grad()
def compute_kid_metrics(fake_pools, real_pools, device="cuda", subsets=50):
    """
    fake_pools[h]: [M_h,3,H,W] float in [0,1] on CPU
    real_pools[h]: [N_h,3,H,W] float in [0,1] on CPU
    horizons are typically 1..4
    """
    kid_results = {"per_horizon": {}, "overall": None}

    horizons = sorted(set(fake_pools.keys()) & set(real_pools.keys()))

    # ---- per-horizon ----
    for h in horizons:
        fake = fake_pools[h].float().clamp(0, 1)
        real = real_pools[h].float().clamp(0, 1)

        fake_u8 = (fake * 255).to(torch.uint8)
        real_u8 = (real * 255).to(torch.uint8)

        subset_size = min(fake_u8.shape[0], real_u8.shape[0])
        if subset_size < 2:
            kid_results["per_horizon"][h] = {"mean": float("nan"), "std": float("nan"), "subset_size": int(subset_size)}
            continue

        kid = KernelInceptionDistance(subset_size=subset_size, subsets=subsets).to(device)
        kid.update(fake_u8.to(device), real=False)
        kid.update(real_u8.to(device), real=True)
        mean, std = kid.compute()

        kid_results["per_horizon"][h] = {
            "mean": float(mean.item()),
            "std": float(std.item()),
            "subset_size": int(subset_size),
        }

    # ---- unweighted overall (mean of per-horizon means) ----
    valid_h = [h for h in horizons if not torch.isnan(torch.tensor(kid_results["per_horizon"][h]["mean"]))]

    if len(valid_h) == 0:
        kid_results["overall_unweighted"] = {"mean": float("nan"), "std": float("nan"), "horizons_used": []}
    else:
        means = [kid_results["per_horizon"][h]["mean"] for h in valid_h]
        stds  = [kid_results["per_horizon"][h]["std"]  for h in valid_h]

        kid_results["overall_unweighted"] = {
            "mean": float(sum(means) / len(means)),
            "std":  float(sum(stds)  / len(stds)),
            "horizons_used": valid_h,
        }

    # ---- overall (concatenate all horizons that have >=2 samples) ----
    fake_all = []
    real_all = []
    for h in horizons:
        if fake_pools[h].shape[0] >= 2 and real_pools[h].shape[0] >= 2:
            fake_all.append(fake_pools[h])
            real_all.append(real_pools[h])

    if not fake_all:
        kid_results["overall"] = {"mean": float("nan"), "std": float("nan"), "subset_size": 0}
        return kid_results

    fake_all = torch.cat(fake_all, dim=0).float().clamp(0, 1)
    real_all = torch.cat(real_all, dim=0).float().clamp(0, 1)

    fake_u8 = (fake_all * 255).to(torch.uint8)
    real_u8 = (real_all * 255).to(torch.uint8)

    subset_size = min(fake_u8.shape[0], real_u8.shape[0])
    if subset_size < 2:
        kid_results["overall"] = {"mean": float("nan"), "std": float("nan"), "subset_size": int(subset_size)}
        return kid_results

    kid = KernelInceptionDistance(subset_size=subset_size, subsets=subsets).to(device)
    kid.update(fake_u8.to(device), real=False)
    kid.update(real_u8.to(device), real=True)
    mean, std = kid.compute()
    kid_results["overall"] = {"mean": float(mean.item()), "std": float(std.item()), "subset_size": int(subset_size)}
    return kid_results

@torch.no_grad()
def compute_psnr_ssim_from_pools(fake_pools, real_pools, device="cuda"):
    """
    Pools keyed by horizon h=1..4. Each value is [N,3,H,W] float in [0,1] on CPU.
    Returns per_horizon + overall (mean of available horizons).
    """
    psnr_m = PeakSignalNoiseRatio(data_range=1.0).to(device)
    ssim_m = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)

    horizons = sorted(set(fake_pools.keys()) & set(real_pools.keys()))
    out = {"per_horizon": {}, "overall": {}}

    psnr_vals = []
    ssim_vals = []

    for h in horizons:
        fake = fake_pools[h].float().clamp(0, 1).to(device)
        real = real_pools[h].float().clamp(0, 1).to(device)

        n = min(fake.shape[0], real.shape[0])
        if n == 0:
            out["per_horizon"][h] = {"psnr": float("nan"), "ssim": float("nan"), "n": 0}
            continue

        fake = fake[:n]
        real = real[:n]

        p = psnr_m(fake, real).item()
        s = ssim_m(fake, real).item()
        out["per_horizon"][h] = {"psnr": float(p), "ssim": float(s), "n": int(n)}

        psnr_vals.append(p)
        ssim_vals.append(s)

    if psnr_vals:
        out["overall"] = {
            "psnr": float(sum(psnr_vals) / len(psnr_vals)),
            "ssim": float(sum(ssim_vals) / len(ssim_vals)),
            "horizons_used": horizons,
        }
    else:
        out["overall"] = {"psnr": float("nan"), "ssim": float("nan"), "horizons_used": []}

    total_n = sum(out["per_horizon"][h]["n"] for h in horizons if out["per_horizon"][h]["n"] > 0)

    if total_n > 0:
        psnr_weighted = 0.0
        ssim_weighted = 0.0
        for h in horizons:
            n = out["per_horizon"][h]["n"]
            if n <= 0:
                continue
            psnr_weighted += out["per_horizon"][h]["psnr"] * n
            ssim_weighted += out["per_horizon"][h]["ssim"] * n

        out["overall_weighted"] = {
            "psnr": float(psnr_weighted / total_n),
            "ssim": float(ssim_weighted / total_n),
            "total_n": int(total_n),
            "horizons_used": horizons,
        }
    else:
        out["overall_weighted"] = {"psnr": float("nan"), "ssim": float("nan"), "total_n": 0, "horizons_used": []}

    return out