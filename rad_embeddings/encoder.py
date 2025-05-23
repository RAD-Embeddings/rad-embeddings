import numpy as np
from dfa import DFA
import gymnasium as gym

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.env_checker import check_env

from utils.utils import dfa2obs
from utils.sb3.logger_callback import LoggerCallback
from utils.sb3.custom_ppo_policy import CustomPPOPolicy
from utils.sb3.dfa_env_features_extractor import DFAEnvFeaturesExtractor
from utils.sb3.dfa_bisim_env_features_extractor import DFABisimEnvFeaturesExtractor
from utils.sb3.dfa_bisim_prob_env_features_extractor import DFABisimProbEnvFeaturesExtractor

class Encoder():
    def __init__(self, load_file: str):
        model = PPO.load(load_file)
        model.set_parameters(load_file)
        for param in model.policy.parameters():
            param.requires_grad = False
        model.policy.eval()
        self.n_tokens = model.policy.features_extractor.n_tokens
        self.obs2rad = model.policy.features_extractor.obs2rad
        self.rad2token = lambda _rad: model.policy.action_net(_rad).argmax(dim=1)

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
        seed: int | None = None,
        n_envs: int = 16
        ):

        save_dir = save_dir[:-1] if save_dir.endswith("/") else save_dir

        # check_env(gym.make(env_id))
        env = make_vec_env(env_id, n_envs=n_envs)

        features_extractor_class = None
        if "DFAEnv" in env_id:
            features_extractor_class = DFAEnvFeaturesExtractor
        elif "DFABisimEnv" in env_id:
            features_extractor_class = DFABisimEnvFeaturesExtractor
        elif "DFABisimProbEnv" in env_id:
            features_extractor_class = DFABisimProbEnvFeaturesExtractor
        else:
            raise ValueError(f"Invalid environment id {env_id}, expected DFAEnv, DFABisimEnv, or DFABisimProbEnv from dfa-gym")

        config = dict(
            policy = CustomPPOPolicy if "Bisim" in env_id else "MlpPolicy",
            env = env,
            learning_rate = 1e-3,
            n_steps = 512,
            batch_size = 1024,
            n_epochs = 2,
            gamma = 0.9,
            gae_lambda = 0.0,
            clip_range = 0.1,
            ent_coef = 0.0,
            vf_coef = 1.0,
            max_grad_norm = 0.5,
            policy_kwargs = dict(
                features_extractor_class = features_extractor_class,
                features_extractor_kwargs = dict(features_dim = 32, n_tokens = env.unwrapped.get_attr("sampler")[0].n_tokens),
                net_arch=dict(pi=[], vf=[]),
                share_features_extractor=True,
            ),
            verbose = 10,
            tensorboard_log = f"{save_dir}/runs/",
            seed = seed
        )

        model = PPO(**config)

        print("Total number of parameters:", sum(p.numel() for p in model.policy.parameters() if p.requires_grad))
        print(model.policy)

        model.learn(1_000_000, callback=LoggerCallback(gamma=config["gamma"]))
        model.save(f"{save_dir}/{id}_{seed}")

