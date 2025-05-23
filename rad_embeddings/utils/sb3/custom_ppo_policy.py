from typing import Callable, Dict, List, Optional, Tuple, Type, Union

from gymnasium import spaces
import torch
from torch import nn

import torch.nn.functional as F
from torch.distributions import Categorical, Normal, Dirichlet

from stable_baselines3 import PPO
from stable_baselines3.common.policies import ActorCriticPolicy

from stable_baselines3.common.distributions import (
    BernoulliDistribution,
    CategoricalDistribution,
    DiagGaussianDistribution,
    Distribution,
    MultiCategoricalDistribution,
    StateDependentNoiseDistribution,
    make_proba_distribution,
)

class CustomPPOPolicy(ActorCriticPolicy):
    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        lr_schedule: Callable[[float], float],
        use_sde=True,
        *args,
        **kwargs,
    ):
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            use_sde=True,
            use_expln=True,
            squash_output=True,
            *args,
            **kwargs,
        )
        self.value_net = NormL2()
        # self.action_net = None
        # self.mu = nn.Linear(64, 10, True)
        # self.std = nn.Linear(64, 10, True)
        # self.softplus = nn.Softplus()

    # def forward(self, obs: torch.Tensor, deterministic: bool = False) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    #     """
    #     Forward pass in all the networks (actor and critic)

    #     :param obs: Observation
    #     :param deterministic: Whether to sample or use deterministic actions
    #     :return: action, value and log probability of the action
    #     """
    #     # Preprocess the observation if needed
    #     features = self.extract_features(obs)
    #     if self.share_features_extractor:
    #         latent_pi, latent_vf = self.mlp_extractor(features)
    #     else:
    #         pi_features, vf_features = features
    #         latent_pi = self.mlp_extractor.forward_actor(pi_features)
    #         latent_vf = self.mlp_extractor.forward_critic(vf_features)
    #     # Evaluate the values for the given observations
    #     values = self.value_net(latent_vf)

    #     # mu = self.mu(latent_pi)
    #     # std = self.softplus(self.std(latent_pi)) + 1e-3
    #     # dist = Normal(mu, std)

    #     # actions = mu if deterministic else dist.rsample()
    #     # log_prob = dist.log_prob(actions).sum(dim=-1)

    #     # print(actions.shape, log_prob.shape)
    #     # print(actions, log_prob)

    #     distribution = self._get_action_dist_from_latent(latent_pi)
    #     print(distribution, type(distribution))
    #     input()
    #     # actions = distribution.get_actions(deterministic=deterministic)
    #     # log_prob = distribution.log_prob(actions)

    #     # print(actions.shape, log_prob.shape)
    #     # print(actions, log_prob)

    #     # input()

    #     actions = actions.reshape((-1, *self.action_space.shape))  # type: ignore[misc]
    #     return actions, values, log_prob

    # def _get_action_dist_from_latent(self, latent_pi: torch.Tensor) -> Distribution:
    # manual_std_layer uses this with sde=False
    #     """
    #     Retrieve action distribution given the latent codes.

    #     :param latent_pi: Latent code for the actor
    #     :return: Action distribution
    #     """
    #     mean_actions = self.action_net(latent_pi)

    #     if isinstance(self.action_dist, DiagGaussianDistribution):
    #         std = self.std(latent_pi)
    #         return self.action_dist.proba_distribution(mean_actions, std)
    #     elif isinstance(self.action_dist, CategoricalDistribution):
    #         # Here mean_actions are the logits before the softmax
    #         return self.action_dist.proba_distribution(action_logits=mean_actions)
    #     elif isinstance(self.action_dist, MultiCategoricalDistribution):
    #         # Here mean_actions are the flattened logits
    #         return self.action_dist.proba_distribution(action_logits=mean_actions)
    #     elif isinstance(self.action_dist, BernoulliDistribution):
    #         # Here mean_actions are the logits (before rounding to get the binary actions)
    #         return self.action_dist.proba_distribution(action_logits=mean_actions)
    #     elif isinstance(self.action_dist, StateDependentNoiseDistribution):
    #         return self.action_dist.proba_distribution(mean_actions, self.log_std, latent_pi)
    #     else:
    #         raise ValueError("Invalid action distribution")

    # def forward(self, obs: torch.Tensor, deterministic: bool = False) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    #     """
    #     Dirichlet-based policy that returns a continuous probability vector.
    #     :return: (action_probs [batch,K], values [batch,1], log_prob [batch])
    #     """
    #     # 1) Feature extraction (same as before)
    #     features = self.extract_features(obs)
    #     if self.share_features_extractor:
    #         latent_pi, latent_vf = self.mlp_extractor(features)
    #     else:
    #         pi_feat, vf_feat = features
    #         latent_pi = self.mlp_extractor.forward_actor(pi_feat)
    #         latent_vf = self.mlp_extractor.forward_critic(vf_feat)

    #     # 2) Critic value
    #     values = self.value_net(latent_vf)

    #     # 3) Compute raw "logits" for concentration
    #     raw = self.action_net(latent_pi)  # [batch, K]

    #     # 4) Turn logits into positive concentration parameters
    #     concentration = F.softplus(raw) + 1e-3
    #     #    ensures α_i > 0 :contentReference[oaicite:0]{index=0}

    #     # 5) Build Dirichlet and sample continuous action_probs
    #     dist         = Dirichlet(concentration)
    #     action_probs = dist.sample()       # a point on the (K−1)-simplex :contentReference[oaicite:1]{index=1}

    #     # 6) For deterministic, take the mode of Dirichlet: (α − 1) / (Σα − K)
    #     if deterministic:
    #         mode = (concentration - 1) / (concentration.sum(dim=-1, keepdim=True) - concentration.size(-1))
    #         action_probs = torch.clamp(mode, min=0.0)  # negative if α<1 → clamp :contentReference[oaicite:2]{index=2}

    #     # 7) Compute log_prob of that sample
    #     log_prob = dist.log_prob(action_probs)  # exact log p(x) on the simplex :contentReference[oaicite:3]{index=3}

    #     return action_probs, values, log_prob

    # def forward(self, obs: torch.Tensor, deterministic: bool = False) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    #     """
    #     Forward pass in all the networks (actor and critic)

    #     :param obs: Observation
    #     :param deterministic: Whether to sample or use deterministic actions
    #     :return: action, value and log probability of the action
    #     """
    #     # Preprocess the observation if needed
    #     features = self.extract_features(obs)
    #     if self.share_features_extractor:
    #         latent_pi, latent_vf = self.mlp_extractor(features)
    #     else:
    #         pi_features, vf_features = features
    #         latent_pi = self.mlp_extractor.forward_actor(pi_features)
    #         latent_vf = self.mlp_extractor.forward_critic(vf_features)
    #     # Evaluate the values for the given observations
    #     values = self.value_net(latent_vf)
    #     # logits = self.action_net(latent_pi)
    #     # actions = F.softmax(logits, dim=-1)
    #     # log_prob, _ = logits.max(dim=-1)
    #     logits = self.action_net(latent_pi)
    #     dist = Categorical(logits=logits)
    #     probs = dist.probs

    #     if deterministic:
    #         discrete_action = torch.argmax(probs, dim=-1)
    #     else:
    #         discrete_action = dist.sample()

    #     log_prob = dist.log_prob(discrete_action)

    #     actions = probs
    #     return actions, values, log_prob

    #     self.mu = nn.Linear(64, 10, True)
    #     self.std = nn.Linear(64, 10, True)
    #     self.softplus = nn.Softplus()

    # def forward(self, obs: torch.Tensor, deterministic: bool = False) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    #     """
    #     Forward pass in all the networks (actor and critic)

    #     :param obs: Observation
    #     :param deterministic: Whether to sample or use deterministic actions
    #     :return: action, value and log probability of the action
    #     """
    #     # Preprocess the observation if needed
    #     features = self.extract_features(obs)
    #     if self.share_features_extractor:
    #         latent_pi, latent_vf = self.mlp_extractor(features)
    #     else:
    #         pi_features, vf_features = features
    #         latent_pi = self.mlp_extractor.forward_actor(pi_features)
    #         latent_vf = self.mlp_extractor.forward_critic(vf_features)
    #     # Evaluate the values for the given observations
    #     values = self.value_net(latent_vf)

    #     mu = 2 * self.mu(latent_pi)
    #     std = self.softplus(self.std(latent_pi)) + 1e-3
    #     dist = Normal(mu, std)
    #     logits = dist.sample()
    #     actions = F.softmax(logits, dim=1)
    #     log_prob = dist.log_prob(logits)
    #     print(actions.shape, log_prob.shape)

    #     distribution = self._get_action_dist_from_latent(latent_pi)
    #     actions = distribution.get_actions(deterministic=deterministic)
    #     log_prob = distribution.log_prob(actions)
    #     actions = actions.reshape((-1, *self.action_space.shape))  # type: ignore[misc]
    #     print(actions.shape, log_prob.shape)
    #     print(actions, torch.exp(actions).sum(dim=1), log_prob)
    #     input()
    #     return actions, values, log_prob

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
