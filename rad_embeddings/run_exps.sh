
# echo "Seed 1 running..."
# uv run python train_encoder.py 1 &> exps_baseline/seed1.txt

# echo "Seed 2 running..."
# uv run python train_encoder.py 2 &> exps_baseline/seed2.txt

# echo "Seed 3 running..."
# uv run python train_encoder.py 3 &> exps_baseline/seed3.txt

# echo "Seed 4 running..."
# uv run python train_encoder.py 4 &> exps_baseline/seed4.txt

# echo "Seed 5 running..."
# uv run python train_encoder.py 5 &> exps_baseline/seed5.txt

echo "Seed 6 running..."
uv run python train_token_env_policy.py 6 &> exps_no_embed/seed6.txt

echo "Seed 7 running..."
uv run python train_token_env_policy.py 7 &> exps_no_embed/seed7.txt

echo "Seed 8 running..."
uv run python train_token_env_policy.py 8 &> exps_no_embed/seed8.txt

echo "Seed 9 running..."
uv run python train_token_env_policy.py 9 &> exps_no_embed/seed9.txt

echo "Seed 10 running..."
uv run python train_token_env_policy.py 10 &> exps_no_embed/seed10.txt
