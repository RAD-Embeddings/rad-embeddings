import torch
import random
import numpy as np
import gymnasium as gym

from encoder import Encoder

from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.env_checker import check_env

if __name__ == "__main__":

    SEED = 42

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    n_envs = 16
    env_id = "DFABisim2Env-v1"
    encoder_id = env_id + "-encoder"
    save_dir = "storage"

    env = gym.make(env_id)
    check_env(env)

    train_env = make_vec_env(env_id, n_envs=n_envs)
    # eval_env = gym.make(env_id)
    eval_env = None

    Encoder.train(train_env=train_env, eval_env=eval_env, save_dir=save_dir, alg="DQN", id=encoder_id, seed=SEED)

    sampler = env.unwrapped.sampler
    encoder = Encoder(load_file=f"{save_dir}/{encoder_id}")

    dfa = sampler.sample()
    print(dfa)

    rad = encoder.dfa2rad(dfa)
    print(rad)

    token = encoder.rad2token(rad)
    print(token)
