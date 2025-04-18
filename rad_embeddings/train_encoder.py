import sys
import torch
import random
import numpy as np

import dfa_gym
from dfa import DFA

from encoder import Encoder

if __name__ == "__main__":

    try:
        SEED = int(sys.argv[1])
    except:
        SEED = 42

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    env_id = "DFABisimEnv-v1"
    encoder_id = env_id + "-encoder"
    save_dir = "storage"

    Encoder.train(env_id=env_id, save_dir=save_dir, alg="PPO", id=encoder_id, seed=SEED)

    # encoder = Encoder(load_file=f"{save_dir}/{encoder_id}")

    # dfa = DFA(
    #     start=0,
    #     inputs=range(10),
    #     label=lambda s: s == 5,
    #     transition=lambda s, a: s + 1 if s == a and s < 5 else s,
    # ).minimize()
    # print(dfa)

    # rad = encoder.dfa2rad(dfa)
    # print(rad)

    # token = encoder.rad2token(rad)
    # print(token)
