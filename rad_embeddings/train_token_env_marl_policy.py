import sys
import torch
import wandb
import token_env
import gymnasium as gym
from encoder import Encoder
from dfa_gym import DFAWrapper, gym2zoo
from stable_baselines3 import PPO
from rad_embeddings.utils import MarlTokenEnvFeaturesExtractor, MarlLoggerCallback, LoggerCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.env_checker import check_env
from dfa_samplers import ReachSampler, ReachAvoidSampler, RADSampler

import supersuit as ss
from pettingzoo.test import api_test, parallel_api_test
from stable_baselines3.common.vec_env.vec_monitor import VecMonitor

from stable_baselines3.common.vec_env.base_vec_env import VecEnv, VecEnvStepReturn, VecEnvWrapper

SEED = int(sys.argv[1])

n_envs = 16
env_id = "TokenEnv-2-agents-fixed-v1"


# env = gym.make(env_id)
env = token_env.TokenEnv(
    n_agents=3,
    size=(5, 5),
    use_fixed_map=True
)
n_tokens = env.unwrapped.n_tokens
n_agents = env.unwrapped.n_agents

reach_avoid_sampler = ReachAvoidSampler(n_tokens=n_tokens, max_size=6, p=None, prob_stutter=1.0)

env = DFAWrapper(env=env, n_agents=env.unwrapped.n_agents, sampler=reach_avoid_sampler, label_f=token_env.TokenEnv.label_f)
env = gym2zoo(env, black_death=True)


# env = ss.black_death_v3(env) # AssertionError: observation sapces for black death must be Box spaces, is Dict('dfa_obs': Box(0, 9, (370,), int64), 'obs': Box(0, 1, (11, 7, 7), uint8))
env = ss.pettingzoo_env_to_vec_env_v1(env)
env = ss.concat_vec_envs_v1(env, n_envs, num_cpus=1, base_class="stable_baselines3")
env = VecMonitor(env)


encoder = Encoder(load_file=f"exps/DFABisimEnv-v1-encoder_{SEED}.zip")

config = dict(
    policy = "MultiInputPolicy",
    env = env,
    n_steps = 128,
    batch_size = 256,
    gamma = 0.99,
    policy_kwargs = dict(
        features_extractor_class=MarlTokenEnvFeaturesExtractor,
        features_extractor_kwargs=dict(n_agents=n_agents, encoder=encoder),
        net_arch=dict(pi=[64, 64, 64], vf=[64, 64]),
        share_features_extractor=True,
        activation_fn=torch.nn.ReLU
    ),
    verbose = 10,
    tensorboard_log = f"exps_marl/runs/"
)


run = wandb.init(
    entity="beyazit-y-berkeley-eecs",
    project="rad-marl",
    config=config,
    sync_tensorboard=True
)

model = PPO(**config)

print("Total number of parameters:", sum(p.numel() for p in model.policy.parameters() if p.requires_grad))
print(model.policy)

logger_callback = MarlLoggerCallback(gamma=config["gamma"])

model.learn(10_000_000, callback=[logger_callback])
model.save(f"exps_marl/token_env_marl_reach_avoid_policy_seed_{SEED}")

wandb.finish()