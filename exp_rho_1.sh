#!/bin/bash

mkdir -p exp_rho_storage

for seed in {2..2}; do
  CUDA_VISIBLE_DEVICES=1 python rad_embeddings/train_policy.py --seed $seed --config config/exp_rho_1.0.yaml
  CUDA_VISIBLE_DEVICES=1 python rad_embeddings/train_policy.py --seed $seed --config config/exp_rho_0.5.yaml
done

for seed in {2..2}; do
  CUDA_VISIBLE_DEVICES=1 python rad_embeddings/train_policy.py --seed $seed --config config/exp_rho_0.0.yaml
done

