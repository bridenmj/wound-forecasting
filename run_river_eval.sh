#!/bin/bash

python scripts/river_eval.py \
    --config "external/river/config.yaml" \
    --ckpt "checkpoints/river/step_20000.pth" \
    --device cuda \
    --batch_size 8 \
    --num_workers 0 \
    --steps 100 \
    --out "results/river_eval.json"
