import gymnasium as gym

from utils.sb3.custom_ppo_policy import CustomPPOPolicy
from utils.sb3.custom_dqn_policy import CustomDQNPolicy
from utils.sb3.dfa_env_features_extractor import DFAEnvFeaturesExtractor
from utils.sb3.dfa_bisim_env_features_extractor import DFABisimEnvFeaturesExtractor

from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.env_checker import check_env

def get_config(env_id, save_dir, alg, seed):
    n_envs = 16
    check_env(gym.make(env_id))
    env = make_vec_env(env_id, n_envs=n_envs)
    assert "DFAEnv" in env_id or "DFABisimEnv" in env_id
    if alg == "DQN":
        return dict(
            policy = CustomDQNPolicy if "Bisim" in env_id else "MlpPolicy",
            env = env,
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
                features_extractor_class = DFABisimEnvFeaturesExtractor if "Bisim" in env_id else DFAEnvFeaturesExtractor,
                features_extractor_kwargs = dict(features_dim = 32, n_tokens = env.unwrapped.get_attr("sampler")[0].n_tokens),
                net_arch=[]
            ),
            verbose = 10,
            tensorboard_log = f"{save_dir}/runs/",
            seed = seed
        )
    elif alg == "PPO":
        return dict(
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
                features_extractor_class = DFABisimEnvFeaturesExtractor if "Bisim" in env_id else DFAEnvFeaturesExtractor,
                features_extractor_kwargs = dict(features_dim = 32, n_tokens = env.unwrapped.get_attr("sampler")[0].n_tokens),
                net_arch=dict(pi=[], vf=[]),
                share_features_extractor=True,
            ),
            verbose = 10,
            tensorboard_log = f"{save_dir}/runs/",
            seed = seed
        )
    else:
        raise NotImplementedError