from typing import Callable, Dict, List, Optional, Tuple, Type, Union

from gymnasium import spaces
import torch
from torch import nn

from stable_baselines3 import PPO
from stable_baselines3.common.policies import ActorCriticPolicy


# class CustomNetwork(nn.Module):
#     """
#     Custom network for policy and value function.
#     It receives as input the features extracted by the features extractor.

#     :param feature_dim: dimension of the features extracted with the features_extractor (e.g. features from a CNN)
#     :param last_layer_dim_pi: (int) number of units for the last layer of the policy network
#     :param last_layer_dim_vf: (int) number of units for the last layer of the value network
#     """

#     def __init__(
#         self,
#         feature_dim: int,
#         last_layer_dim_pi: int = 64,
#         last_layer_dim_vf: int = 64,
#     ):
#         super().__init__()

#         # IMPORTANT:
#         # Save output dimensions, used to create the distributions
#         self.latent_dim_pi = last_layer_dim_pi
#         self.latent_dim_vf = last_layer_dim_vf

#         # Policy network
#         self.policy_net = nn.Sequential(
#             nn.Linear(feature_dim, last_layer_dim_pi), nn.ReLU()
#         )
#         # # Value network
#         # self.value_net = nn.Sequential(
#         #     nn.Linear(feature_dim, last_layer_dim_vf), nn.ReLU()
#         # )

#     def forward(self, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
#         """
#         :return: (torch.Tensor, torch.Tensor) latent_policy, latent_value of the specified network.
#             If all layers are shared, then ``latent_policy == latent_value``
#         """
#         return self.forward_actor(features), self.forward_critic(features)

#     def forward_actor(self, features: torch.Tensor) -> torch.Tensor:
#         return self.policy_net(features)

#     def forward_critic(self, features: torch.Tensor) -> torch.Tensor:

#         feat1 = features[:, :features.shape[1]//2]
#         feat2 = features[:, features.shape[1]//2:]
#         d = nn.functional.cosine_similarity(feat1, feat2, dim=1, eps=1e-8)
#         # return self.value_net(features)
#         return d


class CustomNetwork(nn.Module):
    """
    Custom network for policy and value function.
    It receives as input the features extracted by the features extractor.

    :param feature_dim: dimension of the features extracted with the features_extractor (e.g. features from a CNN)
    :param last_layer_dim_pi: (int) number of units for the last layer of the policy network
    :param last_layer_dim_vf: (int) number of units for the last layer of the value network
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def forward(self, features: torch.Tensor) -> torch.Tensor:

        feat1 = features[:, :features.shape[1]//2]
        feat2 = features[:, features.shape[1]//2:]
        # d = 1 - nn.functional.cosine_similarity(feat1, feat2, dim=1, eps=1e-8)
        # d = torch.norm(feat1 - feat2, p=1, dim=1)
        # return self.value_net(features)
        feat1 = feat1 / torch.norm(feat1, p=2, dim=-1, keepdim=True)
        feat2 = feat2 / torch.norm(feat2, p=2, dim=-1, keepdim=True)
        # Compute the L2 distance between the normalized vectors
        d = torch.norm(feat1 - feat2, p=2, dim=-1)
        return d

class CustomActorCriticPolicy(ActorCriticPolicy):
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
        self.value_net = CustomNetwork()


    # def _build_mlp_extractor(self) -> None:
    #     self.mlp_extractor = CustomNetwork(self.features_dim)

