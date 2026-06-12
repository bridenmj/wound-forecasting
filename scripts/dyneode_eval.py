import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

# ---- repo paths ----
REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
DYNEODE_CODE = REPO_ROOT / "external" / "DyneODE" / "code"

sys.path.insert(0, str(SRC_ROOT))
sys.path.insert(0, str(DYNEODE_CODE))

from shared.data.manifest import create_datasets_FULL
from dyneode.data import DyneODEEvalDataset
from dyneode.model import ODEfunc, load_generator
from dyneode.eval import run_validation_bucketed_dyneode_variable
from metrics.metrics import compute_kid_metrics, compute_psnr_ssim_from_pools


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate DyneODE wound forecasting.")

    p.add_argument("--manifest", required=True, help="Path to manifest JSON.")
    p.add_argument("--latent_root", required=True, help="Root directory containing DyneODE .pt latents.")
    p.add_argument("--image_root", required=True, help="Root directory containing RGB wound images.")
    p.add_argument("--ode_ckpt", required=True, help="Path to trained DyneODE ODE checkpoint.")
    p.add_argument("--stylegan_ckpt", required=True, help="Path to pretrained StyleGAN checkpoint.")
    p.add_argument("--val_id", default=None, help="Held-out pig id, e.g. ID1326.")
    p.add_argument("--style_dim", type=int, default=512)
    p.add_argument("--latent_layers", type=int, default=18)
    p.add_argument("--depth", type=int, default=1)
    p.add_argument("--image_size", type=int, default=256)
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--device", default="cuda")
    p.add_argument("--out", default="results/dyneode_eval.json")

    return p.parse_args()


def main():
    args = parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # create_datasets_FULL expects args.man_pth in your existing helper
    args.man_pth = args.manifest

    _, val_manifest = create_datasets_FULL(args, val_id=args.val_id)

    val_dataset = DyneODEEvalDataset(
        full_seq_manifest=val_manifest,
        latent_base_dir=args.latent_root,
        image_base_dir=args.image_root,
        image_size=args.image_size,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )

    odefunc = ODEfunc(
        dim=args.style_dim,
        depth=args.depth,
    ).to(device)

    state = torch.load(args.ode_ckpt, map_location=device)
    odefunc.load_state_dict(state)
    odefunc.eval()

    generator = load_generator(args).to(device)
    generator.eval()

    fake_pools, real_pools = run_validation_bucketed_dyneode_variable(
        odefunc=odefunc,
        generator=generator,
        val_loader=val_loader,
        device=device,
    )

    for h in [1, 2, 3, 4]:
        print(f"H{h}: fake {fake_pools[h].shape}, real {real_pools[h].shape}")

    kid = compute_kid_metrics(fake_pools, real_pools, device=device)
    psnr_ssim = compute_psnr_ssim_from_pools(fake_pools, real_pools, device=device)

    results = {
        "kid": kid,
        "psnr_ssim": psnr_ssim,
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    print("Saved:", out_path)
    print("KID:", kid)
    print("PSNR/SSIM:", psnr_ssim)


if __name__ == "__main__":
    main()