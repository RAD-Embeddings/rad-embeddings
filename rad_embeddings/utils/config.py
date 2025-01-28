
from utils.custom_network import CustomActorCriticPolicy
from utils.custom_dqn_policy import CustomDQNPolicy
from utils.sb3_dfa_env_features_extractor import DFAEnvFeaturesExtractor

def get_config(alg, train_env, save_dir, seed):
    if alg == "DQN":
        return dict(
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
                features_extractor_kwargs = dict(features_dim = 32, n_tokens = train_env.unwrapped.get_attr("sampler")[0].n_tokens),
                net_arch=[]
            ),
            verbose = 10,
            tensorboard_log = f"{save_dir}/runs/",
            seed = seed
        )
    elif alg == "PPO":
        return dict(
            policy = CustomActorCriticPolicy,
            env = train_env,
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
                features_extractor_class = DFAEnvFeaturesExtractor,
                features_extractor_kwargs = dict(features_dim = 32, n_tokens = train_env.unwrapped.get_attr("sampler")[0].n_tokens),
                net_arch=dict(pi=[], vf=[]),
                share_features_extractor=True,
            ),
            verbose = 10,
            tensorboard_log = f"{save_dir}/runs/",
            seed = seed
        )
    else:
        raise NotImplementedError