import torch
import torch.nn as nn
import torch.nn.functional as F
from rad_embeddings.utils.utils import feature_inds, obs2feat
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from torch_geometric.nn import GATv2Conv

from torch_geometric.data import Data
from torch_geometric.data import Batch

# class WordTransformer(nn.Module):
#     def __init__(self, n_tokens, embed_dim=64, num_heads=1, hidden_dim=128, num_layers=1, dfa_embed_dim=32):
#         super().__init__()
#         self.embedding = nn.Embedding(n_tokens + 1, embed_dim)
#         self.pos_encoding = nn.Parameter(torch.randn(1, 100, embed_dim))  # Max sequence length = 100
#         self.transformer = nn.TransformerEncoder(
#             nn.TransformerEncoderLayer(embed_dim, num_heads, hidden_dim, batch_first=True),
#             num_layers
#         )
#         self.act = nn.Tanh()
#         self.fc = nn.Linear(embed_dim, dfa_embed_dim)  # Project to DFA embedding space

#     def forward(self, x):
#         x = self.embedding(x) + self.pos_encoding[:, :x.shape[1], :]
#         x = self.transformer(x)
#         x = x.mean(dim=1)  # Pool over sequence dimension
#         return self.fc(self.act(x))  # Output DFA embedding

# class WordRNN(nn.Module):
#     def __init__(self, n_tokens):
#         super().__init__()
#         # self.embedding = nn.Embedding(n_tokens + 1, 64) # + 1 is for the empty string
#         self.embedding = nn.Linear(11, 64)
#         self.rnn = nn.RNN(64, 64, num_layers=1, batch_first=True, nonlinearity='tanh')
#         self.fc = nn.Linear(64, 32)
#         self.n_tokens = n_tokens

#     def forward(self, x):
#         x_one_hot = F.one_hot(x, num_classes=self.n_tokens + 1)
#         embedded = self.embedding(x_one_hot.float())  # (batch, seq_len, embed_dim)
#         rnn_out = self.rnn(embedded)[0]  # (batch, seq_len, hidden_dim)
#         summed = rnn_out.sum(dim=1)  # Sum over sequence length
#         return self.fc(summed)  # (batch, 32)

class WordEnvFeaturesExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim, n_tokens, **kwargs):
        super().__init__(observation_space, features_dim)
        self.n_tokens = n_tokens
        # self.rnn = WordRNN(n_tokens=self.n_tokens)
        # self.model = WordTransformer(n_tokens=self.n_tokens)self.input_dim = input_dim
        self.output_dim = features_dim
        self.hidden_dim = kwargs.get("hidden_dim", 64)
        self.num_layers = kwargs.get("num_layers", 8)
        self.n_heads = kwargs.get("n_heads", 4)

        self.linear_in = nn.Linear(12, self.hidden_dim)
        self.conv = GATv2Conv(2*self.hidden_dim, self.hidden_dim, heads=self.n_heads, add_self_loops=True)
        self.activation = nn.Tanh()
        self.g_embed = nn.Linear(self.hidden_dim, self.output_dim)

    def forward(self, obs):
        batch = self.obs2batch(obs)
        feat = batch.feat
        edge_index = batch.edge_index
        B = batch.B
        L = batch.L
        h_0 = self.linear_in(feat.float())
        h = h_0
        for _ in range(L):
            h = self.conv(torch.cat([h, h_0], 1), edge_index).view(-1, self.n_heads, self.hidden_dim).sum(1)
            h = self.activation(h)
        hg = h[torch.arange(B), :]
        z = self.g_embed(hg)
        return z
        

    def obs2batch(self, obs_batch):
        B, _, L = obs_batch.shape
        temp = []
        for obs in obs_batch:
            n, l = obs.shape
            nonzero_word_idx = (obs != 0).any(dim=1).nonzero()[:, 0]
            nonzero_words = obs[nonzero_word_idx]
            m, l = nonzero_words.shape
            one_hot_words = F.one_hot(nonzero_words.long(), num_classes=self.n_tokens + 1).view(m, l, -1)
            feat_batch = F.pad(one_hot_words, (0, 1, 0, 0)) # (M, L, A) to (M, L, A + 1) s.t. +1 in A+1 is the last element
            m, l, f = feat_batch.shape
            master_node_feat = torch.zeros((1, f))
            master_node_feat[:, -1] = 1

            feat = torch.cat([master_node_feat, feat_batch.view(m*l, f)[:, ]], dim=0)

            edge_index = self._get_edge_index(m, l)
            data = Data(feat=feat, edge_index=edge_index.T)
            temp.append(data)
        batch = Batch.from_data_list(temp)
        batch.B = B
        batch.L = L
        return batch

    # def _get_edge_index(self, B, M, L):
    #     block_size = M * L
    #     to_agg_nodes = [[B + i * block_size + j, i] for i in range(B) for j in range(M)]
    #     to_prev_node = [[B + i * block_size + j * L + k + 1, B + i * block_size + j * L + k] for i in range(B) for j in range(M) for k in range(L - 1)]
    #     return torch.tensor(to_agg_nodes + to_prev_node)

    def _get_edge_index(self, M, L):
        to_agg_nodes = [[i, 0] for i in range(1, M * L + 1, L)]
        to_prev_node = [[j + 1, j] for i in range(1, M * L + 1, L) for j in range(i, i + L - 1)]
        return torch.tensor(to_agg_nodes + to_prev_node)
