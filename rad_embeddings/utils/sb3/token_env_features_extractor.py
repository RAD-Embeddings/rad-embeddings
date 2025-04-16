import torch
from torch import nn
from rad_embeddings.model import Model
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from rad_embeddings.utils.utils import feature_inds, obs2feat

class TokenEnvFeaturesExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim, encoder=None):
        super().__init__(observation_space, features_dim)
        self.n_tokens = 10
        in_feat_size = self.n_tokens + len(feature_inds)
        self.model = Model(input_dim=in_feat_size, output_dim=32)
        c, w, h = observation_space["obs"].shape # CxWxH
        self.image_conv = nn.Sequential(
            nn.Conv2d(c, 16, (2, 2)),
            nn.ReLU(),
            nn.Conv2d(16, 32, (2, 2)),
            nn.ReLU(),
            nn.Conv2d(32, 64, (2, 2)),
            nn.ReLU(),
            nn.Flatten()
        )

    def forward(self, dict_obs):
        rad = self.model(obs2feat(dict_obs["dfa_obs"], n_tokens=self.n_tokens))
        obs = self.image_conv(dict_obs["obs"])
        obs = torch.cat((obs, rad), dim=1)
        return obs

# class TokenEnvFeaturesExtractor(BaseFeaturesExtractor):
#     def __init__(self, observation_space, features_dim, encoder):
#         super().__init__(observation_space, features_dim)
#         self.encoder = encoder
#         c, w, h = observation_space["obs"].shape # CxWxH
#         self.image_conv = nn.Sequential(
#             nn.Conv2d(c, 16, (2, 2)),
#             nn.ReLU(),
#             nn.Conv2d(16, 32, (2, 2)),
#             nn.ReLU(),
#             nn.Conv2d(32, 64, (2, 2)),
#             nn.ReLU(),
#             nn.Flatten()
#         )

#     def forward(self, dict_obs):
#         dfa_obs = dict_obs["dfa_obs"]
#         obs = dict_obs["obs"]
#         rad = self.encoder.obs2rad(dfa_obs)
#         obs = self.image_conv(obs)
#         obs = torch.cat((obs, rad), dim=1)
#         return obs
