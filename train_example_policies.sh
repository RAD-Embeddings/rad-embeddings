#!/bin/bash

DEVICE_ID=$1

if [ -z "$DEVICE_ID" ]; then
  echo "Usage: $0 <CUDA_DEVICE_ID>"
  exit 1
fi

mkdir -p example_storage

for seed in {42..42}; do
  echo "Running seed $seed on GPU $DEVICE_ID..."
  CUDA_VISIBLE_DEVICES=$DEVICE_ID python rad_embeddings/train_policy.py \
    --seed $seed \
    --config config/example.yaml
done

