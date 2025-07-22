import torch
from torch import nn
from rad_embeddings.model import Model
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from rad_embeddings.utils.utils import feature_inds, obs2feat

class MarlTokenEnvFeaturesExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, n_agents, encoder):
        super().__init__(observation_space, (observation_space["obs"].shape[1] - 3) * (observation_space["obs"].shape[2] - 3) * 64 + n_agents * encoder.output_dim)
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
        self.obs_embed_size = (w - 3) * (h - 3) * 64

    # def forward(self, dict_obs):
    #     # print(dict_obs.keys())
    #     dfa_obs = dict_obs["dfa_obs"]                     # shape: [48, 188]
    #     other_dfa_obss = dict_obs["other_dfa_obss"]       # shape: [48, 2, 188]
    #     obs = dict_obs["obs"]

    #     # Pass only the [48, 188] dfa_obs to obs2rad
    #     rad = self.encoder.obs2rad(dfa_obs)               # shape: [48, D]

    #     # Reshape other_dfa_obss from [48, 2, 188] → [96, 188] for obs2rad
    #     other_dfa_obss_flat = other_dfa_obss.view(-1, other_dfa_obss.shape[-1])  # shape: [96, 188]
    #     other_rad = self.encoder.obs2rad(other_dfa_obss_flat)                    # shape: [96, D]

    #     # Reshape back to [48, 2 * D] to concatenate per batch item
    #     B, K, D_in = other_dfa_obss.shape  # B=48, K=2, D_in=188
    #     D_out = other_rad.shape[1]        # output dim of obs2rad
    #     other_rad = other_rad.view(B, K * D_out)          # shape: [48, 2 * D]

    #     obs = self.image_conv(obs)                        # shape: [48, ?]
    #     obs = torch.cat((obs, rad, other_rad), dim=1)     # shape: [48, ? + D + 2*D]
    #     return obs

    # def forward(self, dict_obs):
    #     dfa_obs = dict_obs["dfa_obs"]
    #     obs = dict_obs["obs"]
    #     all_zero_mask = torch.all(dfa_obs == 0, dim=-1)  # shape: (b, n)
    #     b, n, l = dfa_obs.shape
    #     rad = self.encoder.obs2rad(dfa_obs.view(b * n, l)).view(b, -1)
    #     obs = self.image_conv(obs)
    #     obs = torch.cat((obs, rad), dim=1)
    #     return obs

    def forward(self, dict_obs):
        dfa_obs = dict_obs["dfa_obs"]  # shape: (b, n, l)
        obs = dict_obs["obs"]

        b, n, l = dfa_obs.shape
        flat_dfa_obs = dfa_obs.view(b * n, l)  # shape: (b*n, l)

        # Identify non-zero rows
        dfa_obs_non_zero_mask = ~(flat_dfa_obs == 0).all(dim=1)  # shape: (b*n,)

        # Allocate full output tensor filled with zeros
        rad = torch.zeros((b * n, self.encoder.output_dim), device=dfa_obs.device)

        # Encode only non-zero rows
        if dfa_obs_non_zero_mask.any():
            encoded = self.encoder.obs2rad(flat_dfa_obs[dfa_obs_non_zero_mask])  # shape: (num_nonzero, D)
            rad[dfa_obs_non_zero_mask] = encoded  # insert into the correct positions

        rad = rad.view(b, -1)  # shape: (b, n * D)

        obs_embed = torch.zeros((b, self.obs_embed_size), device=obs.device)

        flat_obs = obs.view(b, -1)
        obs_non_zero_mask = ~(flat_obs == 0).all(dim=1)
        if obs_non_zero_mask.any():
            encoded = self.image_conv(obs[obs_non_zero_mask])
            obs_embed[obs_non_zero_mask] = encoded

        return torch.cat((obs_embed, rad), dim=1)


    # def forward(self, dict_obs):
    #     dfa_obs = dict_obs["dfa_obs"]
    #     other_dfa_obs = dict_obs["other_dfa_obss"].squeeze()
    #     obs = dict_obs["obs"]
    #     rad = self.encoder.obs2rad(dfa_obs)
    #     other_rad = self.encoder.obs2rad(other_dfa_obs)
    #     obs = self.image_conv(obs)
    #     obs = torch.cat((obs, rad, other_rad), dim=1)
    #     return obs
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
