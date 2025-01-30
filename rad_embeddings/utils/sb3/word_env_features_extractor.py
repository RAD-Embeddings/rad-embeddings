import torch
import torch.nn as nn
from rad_embeddings.utils.utils import feature_inds, obs2feat
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

class WordRNN(nn.Module):
    def __init__(self, n_tokens):
        super().__init__()
        self.embedding = nn.Embedding(n_tokens + 1, 64) # + 1 is for the empty string
        self.rnn = nn.RNN(64, 64, batch_first=True)
        self.fc = nn.Linear(64, 32)

    def forward(self, x):
        embedded = self.embedding(x)  # (batch, seq_len, embed_dim)
        rnn_out, _ = self.rnn(embedded)  # (batch, seq_len, hidden_dim)
        summed = rnn_out.sum(dim=1)  # Sum over sequence length
        out = self.fc(summed)
        return out  # (batch, 32)

class WordEnvFeaturesExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim, n_tokens):
        super().__init__(observation_space, features_dim)
        self.n_tokens = n_tokens
        self.rnn = WordRNN(n_tokens=self.n_tokens)
        self.map = nn.Linear(64, 32)

    def forward(self, obs):
        word = obs["word"]
        prev_dfa = obs["prev_dfa"].squeeze()
        z = self.rnn(word.int())
        temp = torch.cat([z, prev_dfa], dim=1)
        out = self.map(temp)
        return out

