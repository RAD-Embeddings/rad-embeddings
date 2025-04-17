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
        dfa_left = bisim["dfa_left"]
        state_belief_left = bisim["state_belief_left"]

        dfa_right = bisim["dfa_right"]
        state_belief_right = bisim["state_belief_right"]

        rad_left = rad = self.obs2rad(dfa_left, state_belief_left)
        rad_right = rad = self.obs2rad(dfa_right, state_belief_right)

        out = torch.cat([rad_left, rad_right], dim=1)
        return out

    def obs2rad(self, obs, state_belief):
        feat = obs2feat(obs, state_belief, n_tokens=self.n_tokens)
        rad = self.model(feat)
        return rad
