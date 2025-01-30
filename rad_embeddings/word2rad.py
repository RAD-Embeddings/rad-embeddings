import dfa_gym
from encoder import Encoder
import gymnasium as gym
from stable_baselines3 import PPO

from stable_baselines3.common.env_util import make_vec_env

from utils.sb3.word_env_features_extractor import WordEnvFeaturesExtractor

from utils.sb3.custom_ppo_policy import CustomPPOPolicy2

n_envs = 16
env_id = "WordEnv-v1"

env = make_vec_env(env_id, n_envs=n_envs)

model = PPO(
    policy = CustomPPOPolicy2,
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
    verbose = 10,
    policy_kwargs = dict(
        features_extractor_class = WordEnvFeaturesExtractor,
        features_extractor_kwargs = dict(features_dim = 32, n_tokens = env.unwrapped.get_attr("sampler")[0].n_tokens),
        net_arch=dict(pi=[], vf=[]),
        share_features_extractor=True,
    ),
    tensorboard_log = "storage/runs/",
)

# Train the model
timesteps = 10_000_000  # Adjust as needed
print("Total number of parameters:", sum(p.numel() for p in model.policy.parameters() if p.requires_grad))
print(model.policy)
model.learn(total_timesteps=timesteps)

# Save the trained model
model.save("ppo_word_env")

# Close the environment
en.close()

# # Load the trained model and test
# model = PPO.load("ppo_word_env", env=word_env)

# obs, info = word_env.reset()
# for _ in range(1000):
#     action, _ = model.predict(obs)  # Get action from trained policy
#     obs, reward, done, truncated, info = word_env.step(action)
#     print(obs, action, reward)
#     if done:
#         break

# word_env.close()
