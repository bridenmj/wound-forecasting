import torch
from tqdm.auto import tqdm

@torch.no_grad()
def run_validation_bucketed_river(
    model,
    val_loader,
    device="cuda",
    steps=100,
    past_horizon=-1,
    store_u8=False,
):
    """
    Expects val_loader batches of full real sequences:
        batch: [B, T, 3, H, W]
    where T may be fixed at 8 with padding/selection, or variable by loader design.

    Assumes evaluation protocol:
        observe first 4 frames
        generate remaining L-4 frames, where L in {5,6,7,8}

    Returns:
        fake_pools[h], real_pools[h] for h in {1,2,3,4}
        each tensor is [N, 3, H, W] on CPU
    """
    model.eval()

    fake_pools = {1: [], 2: [], 3: [], 4: []}
    real_pools = {1: [], 2: [], 3: [], 4: []}

    for batch in tqdm(val_loader, leave=False):
        # batch should be [B, T, 3, H, W]
        batch = batch.to(device, non_blocking=True)

        B, T, C, H, W = batch.shape

        # If your val loader always returns 8-frame sequences, then infer L=8 for all.
        # If it returns padded sequences, you need true lengths separately.
        #
        # Best case: your val dataset returns (data, seq_len) or metadata.
        # For now assume all in batch share the same full length T.
        L = T

        if L < 5 or L > 8:
            continue

        context = batch[:, :4]          # [B, 4, 3, H, W]
        real_future = batch[:, 4:L]     # [B, L-4, 3, H, W]
        num_tgt = L - 4

        generated = model.generate_frames(
            observations=context,
            num_frames=num_tgt,
            steps=steps,
            past_horizon=past_horizon,
            verbose=False,
        )  # [B, 4+num_tgt, 3, H, W]

        # Keep only generated future frames
        fake_future = generated[:, 4:]  # [B, num_tgt, 3, H, W]

        for j in range(num_tgt):
            h = j + 1  # horizon 1..4

            fake_j = fake_future[:, j].detach().cpu()
            real_j = real_future[:, j].detach().cpu()

            if store_u8:
                fake_j = (fake_j.clamp(0, 1) * 255).to(torch.uint8)
                real_j = (real_j.clamp(0, 1) * 255).to(torch.uint8)

            fake_pools[h].append(fake_j)
            real_pools[h].append(real_j)

    out_dtype = torch.uint8 if store_u8 else torch.float32
    for h in (1, 2, 3, 4):
        fake_pools[h] = torch.cat(fake_pools[h], dim=0) if len(fake_pools[h]) else torch.empty(0, dtype=out_dtype)
        real_pools[h] = torch.cat(real_pools[h], dim=0) if len(real_pools[h]) else torch.empty(0, dtype=out_dtype)

    return fake_pools, real_pools

@torch.no_grad()
def run_validation_bucketed_river_variable(
    model,
    val_loader,
    device="cuda",
    steps=100,
    past_horizon=-1,
    store_u8=False,
):
    model.eval()

    fake_pools = {1: [], 2: [], 3: [], 4: []}
    real_pools = {1: [], 2: [], 3: [], 4: []}

    for batch in tqdm(val_loader, leave=False):
        # expected:
        # data: [B, T_max, 3, H, W]
        # seq_len: [B]
        data, seq_len = batch

        data = data.to(device, non_blocking=True)
        seq_len = seq_len.to(device)

        for L in (5, 6, 7, 8):
            idx = (seq_len == L).nonzero(as_tuple=True)[0]
            if idx.numel() == 0:
                continue

            x = data.index_select(0, idx)[:, :L]   # [B_L, L, 3, H, W]
            context = x[:, :4]
            real_future = x[:, 4:L]
            num_tgt = L - 4

            generated = model.generate_frames(
                observations=context,
                num_frames=num_tgt,
                steps=steps,
                past_horizon=past_horizon,
                verbose=False,
            )
            fake_future = generated[:, 4:]

            for j in range(num_tgt):
                h = j + 1
                fake_j = fake_future[:, j].detach().cpu()
                real_j = real_future[:, j].detach().cpu()

                if store_u8:
                    fake_j = (fake_j.clamp(0, 1) * 255).to(torch.uint8)
                    real_j = (real_j.clamp(0, 1) * 255).to(torch.uint8)

                fake_pools[h].append(fake_j)
                real_pools[h].append(real_j)

    out_dtype = torch.uint8 if store_u8 else torch.float32
    for h in (1, 2, 3, 4):
        fake_pools[h] = torch.cat(fake_pools[h], dim=0) if len(fake_pools[h]) else torch.empty(0, dtype=out_dtype)
        real_pools[h] = torch.cat(real_pools[h], dim=0) if len(real_pools[h]) else torch.empty(0, dtype=out_dtype)

    return fake_pools, real_pools


import torch
from torch.utils.data import Dataset

class EvalWrapper(Dataset):
    def __init__(self, base_dataset):
        self.base = base_dataset

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        data, raw_frames = self.base[idx]   # base must use return_raw_frames=True
        seq_len = raw_frames.shape[0]
        return data, seq_len