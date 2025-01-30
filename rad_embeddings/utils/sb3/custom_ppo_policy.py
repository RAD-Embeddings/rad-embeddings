from typing import Callable, Dict, List, Optional, Tuple, Type, Union

from gymnasium import spaces
import torch
from torch import nn

import numpy as np

from stable_baselines3 import PPO
from stable_baselines3.common.policies import ActorCriticPolicy

class CustomPPOPolicy2(ActorCriticPolicy):
    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        lr_schedule: Callable[[float], float],
        *args,
        **kwargs,
    ):
        super().__init__(
            observation_space,
            spaces.Box(low=-1, high=1, shape=(1, 32), dtype=np.float32),
            lr_schedule,
            *args,
            **kwargs,
        )
        self.action_net = nn.Identity()
        self.value_net = nn.Linear(in_features=32, out_features=1, bias=True)

    def forward(self, obs, deterministic: bool = False):
        features = self.extract_features(obs)
        hidden = features[:, 32:]
        features = features[:, :32]
        if self.share_features_extractor:
            latent_pi, latent_vf = self.mlp_extractor(features)
        else:
            pi_features, vf_features = features
            latent_pi = self.mlp_extractor.forward_actor(pi_features)
            latent_vf = self.mlp_extractor.forward_critic(vf_features)
        # Evaluate the values for the given observations
        print(latent_pi.shape)
        input()
        values = self.value_net(latent_vf)
        distribution = self._get_action_dist_from_latent(latent_pi)
        actions = distribution.get_actions(deterministic=deterministic)
        log_prob = distribution.log_prob(actions)
        actions = actions.reshape((-1, *self.action_space.shape))  # type: ignore[misc]
        return actions, values, log_prob



class CustomPPOPolicy(ActorCriticPolicy):
    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        lr_schedule: Callable[[float], float],
        *args,
        **kwargs,
    ):
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            *args,
            **kwargs,
        )
        self.value_net = NormL2()

class NormL2(nn.Module):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        feat1 = features[:, :features.shape[1]//2]
        feat2 = features[:, features.shape[1]//2:]

        feat1 = feat1 / torch.norm(feat1, p=2, dim=-1, keepdim=True)
        feat2 = feat2 / torch.norm(feat2, p=2, dim=-1, keepdim=True)
        d = torch.norm(feat1 - feat2, p=2, dim=-1)
        return d

class CosDist(nn.Module):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        feat1 = features[:, :features.shape[1]//2]
        feat2 = features[:, features.shape[1]//2:]
        d = 1 - nn.functional.cosine_similarity(feat1, feat2, dim=1, eps=1e-8)
        return d
