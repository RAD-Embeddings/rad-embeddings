#!/bin/bash

DEVICE_ID=$1
START_SEED=$2
END_SEED=$3

if [ -z "$DEVICE_ID" ] || [ -z "$START_SEED" ] || [ -z "$END_SEED" ]; then
  echo "Usage: $0 <CUDA_DEVICE_ID> <START_SEED> <END_SEED>"
  exit 1
fi

export CUDA_VISIBLE_DEVICES=$DEVICE_ID

mkdir -p storage

for seed in $(seq $START_SEED $END_SEED); do
  python train.py \
    --seed $seed \
    --wandb \
    --log storage/log_${seed}.csv &> storage/out_$seed.txt
done

