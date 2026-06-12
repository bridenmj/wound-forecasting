#!/bin/bash

python scripts/dyneode_eval.py \
    --manifest "/content/drive/MyDrive/Heals_Winter_24/JMIR Heals/manifest.json" \
    --latent_root "data/DyneODE/latents" \
    --image_root "/content/512x512" \
    --ode_ckpt "checkpoints/DyneODE/epoch_4000.pth" \
    --stylegan_ckpt "checkpoints/DyneODE/jmir_iteration_30000.pt" \
    --val_id ID1326 \
    --depth 5 \
    --style_dim 512 \
    --image_size 256 \
    --out results/dyneode_eval.json
