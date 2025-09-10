#!/bin/bash

mkdir -p exp_on_rew_frac

for seed in {0..2}; do
  CUDA_VISIBLE_DEVICES=0 python rad_embeddings/train_policy.py --seed $seed --config config/exp_on_rew_frac_0.0.yaml &> out_exp_on_rew_frac_0.0.txt
  CUDA_VISIBLE_DEVICES=0 python rad_embeddings/train_policy.py --seed $seed --config config/exp_on_rew_frac_0.2.yaml &> out_exp_on_rew_frac_0.2.txt
  CUDA_VISIBLE_DEVICES=0 python rad_embeddings/train_policy.py --seed $seed --config config/exp_on_rew_frac_0.4.yaml &> out_exp_on_rew_frac_0.4.txt
done

