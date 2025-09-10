#!/bin/bash

mkdir -p exp_rho_storage

for seed in {0..4}; do
  CUDA_VISIBLE_DEVICES=0 python rad_embeddings/train_policy.py --seed $seed --config config/exp_rho_0.5.yaml &> out.txt
done

for seed in {0..4}; do
  CUDA_VISIBLE_DEVICES=0 python rad_embeddings/train_policy.py --seed $seed --config config/exp_rho_1.0.yaml &> out.txt
done

for seed in {0..4}; do
  CUDA_VISIBLE_DEVICES=0 python rad_embeddings/train_policy.py --seed $seed --config config/exp_rho_0.0.yaml &> out.txt
done

