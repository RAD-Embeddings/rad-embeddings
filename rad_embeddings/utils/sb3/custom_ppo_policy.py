from typing import Callable, Dict, List, Optional, Tuple, Type, Union

from gymnasium import spaces
import torch
from torch import nn

from stable_baselines3 import PPO
from stable_baselines3.common.policies import ActorCriticPolicy

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
