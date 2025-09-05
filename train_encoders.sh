#!/bin/bash

DEVICE_ID=$1

if [ -z "$DEVICE_ID" ]; then
  echo "Usage: $0 <CUDA_DEVICE_ID>"
  exit 1
fi

mkdir -p encoder_storage

for seed in {0..9}; do
  echo "Running seed $seed on GPU $DEVICE_ID..."
  CUDA_VISIBLE_DEVICES=$DEVICE_ID python rad_embeddings/train_encoder.py \
    --seed $seed \
    --wandb \
    --log encoder_storage/log_${seed}.csv
done

