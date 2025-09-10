#!/bin/bash

mkdir -p exp_on_rew_frac

for seed in {0..2}; do
  CUDA_VISIBLE_DEVICES=1 python rad_embeddings/train_policy.py --seed $seed --config config/exp_on_rew_frac_0.6.yaml &> out_exp_on_rew_frac_0.6.txt
  CUDA_VISIBLE_DEVICES=1 python rad_embeddings/train_policy.py --seed $seed --config config/exp_on_rew_frac_0.8.yaml &> out_exp_on_rew_frac_0.8.txt
  CUDA_VISIBLE_DEVICES=1 python rad_embeddings/train_policy.py --seed $seed --config config/exp_on_rew_frac_1.0.yaml &> out_exp_on_rew_frac_1.0.txt
done

