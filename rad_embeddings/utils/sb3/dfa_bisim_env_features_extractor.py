import torch
from rad_embeddings.model import Model
from rad_embeddings.utils.utils import feature_inds, obs2feat
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

class DFABisimEnvFeaturesExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim, n_tokens, reparam=False, model_cls=Model):
        super().__init__(observation_space, features_dim*2)
        in_feat_size = n_tokens + len(feature_inds)
        self.model = model_cls(in_feat_size, features_dim, reparam)
        self.n_tokens = n_tokens

    def forward(self, bisim):
        obs = torch.cat(torch.split(bisim, bisim.shape[1]//2, dim=1), dim=0)
        rad = self.obs2rad(obs)
        out = torch.cat(torch.split(rad, rad.shape[0]//2, dim=0), dim=1)
        return out

    def obs2rad(self, obs):
        feat = obs2feat(obs, n_tokens=self.n_tokens)
        rad = self.model(feat)
        return rad
