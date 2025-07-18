import torch
from torch import nn
from rad_embeddings.model import Model
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from rad_embeddings.utils.utils import feature_inds, obs2feat

class MarlTokenEnvFeaturesExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim, encoder):
        super().__init__(observation_space, features_dim)
        self.encoder = encoder
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
        dfa_obs = dict_obs["dfa_obs"]
        obs = dict_obs["obs"]
        rad = self.encoder.obs2rad(dfa_obs)
        obs = self.image_conv(obs)
        obs = torch.cat((obs, rad), dim=1)
        return obs
    # def forward(self, dict_obs):
    #     dfa_obs = dict_obs["dfa_obs"]
    #     obs = dict_obs["obs"]

    #     # Find non-zero entries (assuming last dimension encodes the vector)
    #     nonzero_mask = (dfa_obs != 0).any(dim=1)
        
    #     # Initialize rad tensor with zeros
    #     rad = torch.zeros(size=(dfa_obs.shape[0], 32), device=dfa_obs.device, dtype=dfa_obs.dtype)

    #     # Only pass non-zero entries through the encoder
    #     if nonzero_mask.any():
    #         encoded = self.encoder.obs2rad(dfa_obs[nonzero_mask])
    #         rad[nonzero_mask] = encoded

    #     # Process obs and concatenate
    #     obs = self.image_conv(obs)
    #     obs = torch.cat((obs, rad), dim=1)
    #     return obs
