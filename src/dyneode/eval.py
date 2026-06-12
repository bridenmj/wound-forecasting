import torch
from torchdiffeq import odeint
from metrics.metrics import compute_kid_metrics, compute_psnr_ssim_from_pools

@torch.no_grad()
def run_validation_bucketed_dyneode_variable(
    odefunc,
    generator,
    val_loader,
    device="cuda",
):
    odefunc.eval()
    generator.eval()

    fake_pools = {1: [], 2: [], 3: [], 4: []}
    real_pools = {1: [], 2: [], 3: [], 4: []}

    for batch in val_loader:
        latents = batch["latents"].to(device)   # [1, T, 18, 512] or [1, T, 14, 512]
        t_steps = batch["t_steps"].to(device)   # [1, T]
        gt_images = batch["images"].to(device)  # [1, T, 3, H, W] in [0,1]
        seq_len = int(batch["seq_len"].item())

        if seq_len < 5 or seq_len > 8:
            continue

        z = latents[0, :seq_len]      # [T, 18, 512] or [T, 14, 512]
        ts = t_steps[0, :seq_len]     # [T]
        imgs = gt_images[0, :seq_len] # [T, 3, H, W]

        pred_traj = odeint(
            odefunc,
            z[0],          # initial latent
            ts,            # full time grid
            method="rk4",
            atol=1e-3,
            rtol=1e-3,
        )  # [T, 18, 512] or [T, 14, 512]

        for j in range(seq_len - 4):
            h = j + 1

            fake_img = generator(
                [pred_traj[4 + j: 5 + j]],
                input_is_latent=True,
                randomize_noise=False,
                return_latents=False,
            )[0]

            fake_img = fake_img.detach().cpu().clamp(-1, 1)
            fake_img = (fake_img + 1) / 2          # [1, 3, H, W] -> [0,1]
            fake_img = fake_img.squeeze(0)         # [3, H, W]

            real_img = imgs[4 + j].detach().cpu().clamp(0, 1)  # actual GT image

            fake_pools[h].append(fake_img.unsqueeze(0))
            real_pools[h].append(real_img.unsqueeze(0))

    for h in (1, 2, 3, 4):
        fake_pools[h] = torch.cat(fake_pools[h], dim=0) if fake_pools[h] else torch.empty(0, 3, 256, 256)
        real_pools[h] = torch.cat(real_pools[h], dim=0) if real_pools[h] else torch.empty(0, 3, 256, 256)

    return fake_pools, real_pools