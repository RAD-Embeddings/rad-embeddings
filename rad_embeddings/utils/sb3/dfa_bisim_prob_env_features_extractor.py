import torch
from rad_embeddings.model import Model
from rad_embeddings.utils.utils import feature_inds, obs2feat
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

class DFABisimProbEnvFeaturesExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim, n_tokens, model_cls=Model):
        super().__init__(observation_space, features_dim*2)
        in_feat_size = n_tokens + len(feature_inds)
        self.model = model_cls(in_feat_size, features_dim)
        self.n_tokens = n_tokens

    def forward(self, obs):
        dfa_left = obs["dfa_left"]
        dfa_left_state_belief = obs["dfa_left_state_belief"]
        dfa_right = obs["dfa_right"]
        dfa_right_state_belief = obs["dfa_right_state_belief"]
        feat_left = obs2feat(dfa_left, state_belief=dfa_left_state_belief, n_tokens=self.n_tokens)
        feat_right = obs2feat(dfa_right, state_belief=dfa_right_state_belief, n_tokens=self.n_tokens)
        rad_left = self.model(feat_left)
        rad_right = self.model(feat_right)
        out = torch.cat([rad_left, rad_right], dim=1)
        return out
