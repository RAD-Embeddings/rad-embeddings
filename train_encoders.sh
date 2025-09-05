#!/bin/bash

for seed in {0..9}; do
  python rad_embeddings/train_encoder.py \
    --seed $seed \
    --wandb \
    --log encoder_storage/log_${seed}.csv
done

