import sys
import torch
import token_env
import gymnasium as gym
from encoder import Encoder
from dfa_gym import DFAWrapper
from stable_baselines3 import PPO
from rad_embeddings.utils import TokenEnvFeaturesExtractor, LoggerCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.env_checker import check_env
from dfa_samplers import ReachSampler, ReachAvoidSampler, RADSampler

SEED = int(sys.argv[1])

n_envs = 16
env_id = "TokenEnv-v1-fixed"

env = gym.make(env_id)
check_env(env)
n_tokens = env.unwrapped.n_tokens

# reach_sampler = ReachSampler(n_tokens=n_tokens, max_size=4, prob_stutter=1.0)
reach_avoid_sampler = ReachAvoidSampler(n_tokens=n_tokens, max_size=6, p=None, prob_stutter=1.0)

# env_kwargs = dict(env_id=env_id, sampler=RADSampler(n_tokens=n_tokens), label_f=token_env.TokenEnv.label_f)
# env_kwargs = dict(env_id=env_id, sampler=ReachAvoidSampler(n_tokens=n_tokens, max_size=4, prob_stutter=1.0), label_f=token_env.TokenEnv.label_f)

env_kwargs = dict(env_id=env_id, sampler=reach_avoid_sampler, label_f=token_env.TokenEnv.label_f)

env = make_vec_env(DFAWrapper, env_kwargs=env_kwargs, n_envs=n_envs)

encoder = Encoder(load_file=f"exps_baseline/DFAEnv-v1-encoder_{SEED}.zip")

config = dict(
    policy = "MultiInputPolicy",
    env = env,
    n_steps = 128,
    batch_size = 256,
    gamma = 0.99,
    policy_kwargs = dict(
        features_extractor_class=TokenEnvFeaturesExtractor,
        features_extractor_kwargs=dict(features_dim=1056, encoder=encoder),
        net_arch=dict(pi=[64, 64, 64], vf=[64, 64]),
        share_features_extractor=True,
        activation_fn=torch.nn.ReLU
    ),
    verbose = 10,
    tensorboard_log = f"exps_baseline/token_env/runs/"
)

model = PPO(**config)

print("Total number of parameters:", sum(p.numel() for p in model.policy.parameters() if p.requires_grad))
print(model.policy)

logger_callback = LoggerCallback(gamma=config["gamma"])

model.learn(1_000_000, callback=[logger_callback])
model.save(f"exps_baseline/token_env/token_env_reach_avoid_policy_seed{SEED}")

