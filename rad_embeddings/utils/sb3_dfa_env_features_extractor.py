import torch
from rad_embeddings.model import Model
from rad_embeddings.utils.utils import feature_inds, obs2feat, bisim2feat
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

class DFAEnvFeaturesExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim, model_cls=Model, n_tokens=10):
        super().__init__(observation_space, features_dim*2)
        in_feat_size = n_tokens + len(feature_inds)
        self.model = model_cls(in_feat_size, features_dim)
        self.n_tokens = n_tokens

    # def forward(self, obs):
    #     return self.model(obs2feat(obs, n_tokens=self.n_tokens))

    def forward(self, obs):
        feat1, feat2 = bisim2feat(obs, n_tokens=self.n_tokens)
        rad1 = self.model(feat1)
        rad2 = self.model(feat2)
        out = torch.cat([rad1, rad2], dim=1)
        return out
