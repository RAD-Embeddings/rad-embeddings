import torch
import random
import numpy as np
from dfa import DFA
from dfa_gym import DFAEnv, DFABisimEnv, DFABisim2Env
from stable_baselines3 import PPO
from utils.dqn import DQN
from rad_embeddings.utils import dfa2obs, DFAEnvFeaturesExtractor, LoggerCallback
import gymnasium as gym
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnNoModelImprovement

from utils.config import get_config

class Encoder():
    def __init__(self, load_file: str):
        model = PPO.load(load_file)
        model.set_parameters(load_file)
        for param in model.policy.parameters():
            param.requires_grad = False
        model.policy.eval()
        self.obs2rad = model.policy.features_extractor
        self.rad2token = lambda _rad: model.policy.action_net(_rad).argmax(dim=1)
        self.n_tokens = self.obs2rad.n_tokens

    def dfa2rad(self, dfa: DFA) -> np.array:
        assert len(dfa.inputs) == self.n_tokens
        obs = dfa2obs(dfa)
        rad = self.obs2rad(obs)
        return rad

    @staticmethod
    def train(
        env_id: str,
        save_dir: str,
        alg: str,
        id: str = "rad",
        seed: int | None = None
        ):
        assert alg == "PPO" or alg == "DQN"
        save_dir = save_dir[:-1] if save_dir.endswith("/") else save_dir
        config = get_config(env_id, save_dir, alg, seed)
        model = PPO(**config) if alg == "PPO" else DQN(**config)

        print("Total number of parameters:", sum(p.numel() for p in model.policy.parameters() if p.requires_grad))
        print(model.policy)

        model.learn(1_000_000, callback=LoggerCallback(gamma=config["gamma"]))
        model.save(f"{save_dir}/{id}")

