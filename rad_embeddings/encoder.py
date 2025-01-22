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

from utils.custom_network import CustomActorCriticPolicy
from utils.custom_dqn_policy import CustomDQNPolicy

SEED = 42

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
    def train(n_tokens: int, train_env: gym.Env, save_dir: str, eval_env: gym.Env | None = None, id: str = "rad"):
        save_dir = save_dir[:-1] if save_dir.endswith("/") else save_dir
        config = dict(
            policy = CustomDQNPolicy,
            env = train_env,
            learning_rate = 1e-3,
            buffer_size = 100_000,
            learning_starts = 10_000,
            batch_size = 1024,
            tau = 1.0,
            gamma = 0.9,
            train_freq = 1,
            gradient_steps = 1,
            target_update_interval = 10_000,
            exploration_fraction = 0.0,
            exploration_initial_eps = 0.0,
            exploration_final_eps = 0.0,
            max_grad_norm = 10,
            policy_kwargs = dict(
                features_extractor_class = DFAEnvFeaturesExtractor,
                features_extractor_kwargs = dict(features_dim=32, n_tokens=n_tokens),
                net_arch=[]
            ),
            verbose = 10,
            tensorboard_log = f"{save_dir}/runs/",
            seed=SEED
        )
        # dqn_old_config = dict(
        #     policy = CustomDQNPolicy,
        #     env = train_env,
        #     learning_rate = 1e-3,
        #     buffer_size = 100_000,
        #     learning_starts = 100_000,
        #     batch_size = 1024,
        #     tau = 1.0,
        #     gamma = 0.9,
        #     train_freq = 10,
        #     gradient_steps = 1,
        #     target_update_interval = 10_000,
        #     exploration_fraction = 0.1,
        #     exploration_initial_eps = 1.0,
        #     exploration_final_eps = 0.05,
        #     max_grad_norm = 10,
        #     policy_kwargs = dict(
        #         features_extractor_class = DFAEnvFeaturesExtractor,
        #         features_extractor_kwargs = dict(features_dim=32, n_tokens=n_tokens),
        #         net_arch=[]
        #     ),
        #     verbose = 10,
        #     tensorboard_log = f"{save_dir}/runs/",
        #     seed=SEED
        # )
        # config = dict(
        #     policy = CustomActorCriticPolicy,
        #     env = train_env,
        #     learning_rate = 1e-3,
        #     n_steps = 512,
        #     batch_size = 1024,
        #     n_epochs = 2,
        #     gamma = 0.9,
        #     gae_lambda = 0.0,
        #     clip_range = 0.1,
        #     ent_coef = 0.0,
        #     vf_coef = 1.0,
        #     max_grad_norm = 0.5,
        #     policy_kwargs = dict(
        #         features_extractor_class = DFAEnvFeaturesExtractor,
        #         features_extractor_kwargs = dict(features_dim=32, n_tokens=n_tokens),
        #         net_arch=dict(pi=[], vf=[]),
        #         share_features_extractor=True,
        #     ),
        #     verbose = 10,
        #     tensorboard_log = f"{save_dir}/runs/",
        #     seed=SEED
        # )

        model = DQN(**config)
        # model = PPO(**config)

        print("Total number of parameters:", sum(p.numel() for p in model.policy.parameters() if p.requires_grad))
        print(model.policy)

        # callback_list = []
        # logger_callback = LoggerCallback(gamma=config["gamma"])
        # callback_list.append(logger_callback)
        # if eval_env is not None:
        #     stop_train_callback = StopTrainingOnNoModelImprovement(max_no_improvement_evals=10, min_evals=20, verbose=1)
        #     eval_callback = EvalCallback(eval_env, eval_freq=1000, callback_after_eval=stop_train_callback, verbose=1)
        #     callback_list.append(eval_callback)
        # model.learn(1_000_000, callback=callback_list)
        # model.save(f"{save_dir}/{id}")
        logger_callback = LoggerCallback(gamma=config["gamma"])
        model.learn(1_000_000, callback=logger_callback)
        model.save(f"{save_dir}/{id}")

if __name__ == "__main__":
    from stable_baselines3.common.env_util import make_vec_env
    from stable_baselines3.common.env_checker import check_env

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    n_envs = 16
    env_id = "DFABisim2Env-v1"
    encoder_id = env_id + "-encoder"
    save_dir = "storage"

    env = gym.make(env_id)
    check_env(env)

    n_tokens = env.unwrapped.sampler.n_tokens

    train_env = make_vec_env(env_id, n_envs=n_envs)
    # eval_env = gym.make(env_id)
    eval_env = None

    Encoder.train(n_tokens=n_tokens, train_env=train_env, eval_env=eval_env, save_dir=save_dir, id=encoder_id)

    sampler = env.unwrapped.sampler
    encoder = Encoder(load_file=f"{save_dir}/{encoder_id}")

    dfa = sampler.sample()
    print(dfa)

    rad = encoder.dfa2rad(dfa)
    print(rad)

    token = encoder.rad2token(rad)
    print(token)
