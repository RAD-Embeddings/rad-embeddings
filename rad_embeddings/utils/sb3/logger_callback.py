import numpy as np
from collections import deque
from stable_baselines3.common.callbacks import BaseCallback

class LoggerCallback(BaseCallback):
    def __init__(self, gamma=0.9):
        super().__init__()
        self.gamma = gamma
        self.n = 100
        self.rewards = deque(maxlen=self.n)
        self.discounted_rewards = deque(maxlen=self.n)
        self.episode_lengths = deque(maxlen=self.n)

    def _on_step(self):
        # Log scalar value (here a random variable)
        dones = self.locals["dones"]
        infos = self.locals["infos"]

        for idx, done in enumerate(dones):
            info = infos[idx]
            if (
                done
                and info.get("episode") is not None
            ):
                episode_info = info["episode"]
                reward = info["episode"]['r']
                episode_length = info["episode"]['l']
                t = episode_length - 1
                discounted_reward = reward * (self.gamma ** t)
                self.rewards.append(reward)
                self.discounted_rewards.append(discounted_reward)
                self.episode_lengths.append(episode_length)

        return True

    def _on_rollout_end(self):
        ep_len_min = np.min(self.episode_lengths)
        ep_len_max = np.max(self.episode_lengths)
        ep_len_std = np.std(self.episode_lengths)
        ep_rew_min = np.min(self.rewards)
        ep_rew_max = np.max(self.rewards)
        ep_rew_std = np.std(self.rewards)
        ep_disc_rew_mean = np.mean(self.discounted_rewards)
        self.logger.record("rollout/ep_len_min", ep_len_min)
        self.logger.record("rollout/ep_len_max", ep_len_max)
        self.logger.record("rollout/ep_len_std", ep_len_std)
        self.logger.record("rollout/ep_rew_min", ep_rew_min)
        self.logger.record("rollout/ep_rew_max", ep_rew_max)
        self.logger.record("rollout/ep_rew_std", ep_rew_std)
        self.logger.record("rollout/ep_rew_disc_mean", ep_disc_rew_mean)
