#!/bin/bash

for seed in {0..9}; do
  python train_encoder.py \
    --seed $seed \
    --wandb \
    --log encoder_storage/log_${seed}.csv
done

