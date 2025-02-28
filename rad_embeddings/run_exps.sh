
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

echo "Seed 1 running..."
uv run python train_token_env_policy.py 1 &> exps_baseline/token_env/seed1.txt

echo "Seed 2 running..."
uv run python train_token_env_policy.py 2 &> exps_baseline/token_env/seed2.txt

echo "Seed 3 running..."
uv run python train_token_env_policy.py 3 &> exps_baseline/token_env/seed3.txt

echo "Seed 4 running..."
uv run python train_token_env_policy.py 4 &> exps_baseline/token_env/seed4.txt

echo "Seed 5 running..."
uv run python train_token_env_policy.py 5 &> exps_baseline/token_env/seed5.txt
