import torch
from rad_embeddings.model import Model
from rad_embeddings.utils.utils import feature_inds, obs2feat
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

class DFAEnvFeaturesExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim, n_tokens, model_cls=Model):
        super().__init__(observation_space, features_dim)
        in_feat_size = n_tokens + len(feature_inds)
        self.model = model_cls(in_feat_size, features_dim)
        self.n_tokens = n_tokens

    def forward(self, obs):
        return self.model(obs2feat(obs, n_tokens=self.n_tokens))

    def obs2rad(self, obs):
        return self.forward(obs)
