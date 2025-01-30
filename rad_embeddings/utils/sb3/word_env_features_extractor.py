import torch
import torch.nn as nn
from rad_embeddings.utils.utils import feature_inds, obs2feat
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

class WordTransformer(nn.Module):
    def __init__(self, n_tokens, embed_dim=64, num_heads=1, hidden_dim=128, num_layers=1, dfa_embed_dim=32):
        super().__init__()
        self.embedding = nn.Embedding(n_tokens + 1, embed_dim)
        self.pos_encoding = nn.Parameter(torch.randn(1, 100, embed_dim))  # Max sequence length = 100
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(embed_dim, num_heads, hidden_dim, batch_first=True),
            num_layers
        )
        self.act = nn.Tanh()
        self.fc = nn.Linear(embed_dim, dfa_embed_dim)  # Project to DFA embedding space

    def forward(self, x):
        x = self.embedding(x) + self.pos_encoding[:, :x.shape[1], :]
        x = self.transformer(x)
        x = x.mean(dim=1)  # Pool over sequence dimension
        return self.fc(self.act(x))  # Output DFA embedding

class WordRNN(nn.Module):
    def __init__(self, n_tokens):
        super().__init__()
        self.embedding = nn.Embedding(n_tokens + 1, 64) # + 1 is for the empty string
        self.rnn = nn.RNN(64, 64, num_layers=10, batch_first=True, nonlinearity='tanh')
        self.fc1 = nn.Linear(64*2, 64)
        self.fc2 = nn.Linear(64, 32)

    def forward(self, x, prev_hidden):
        embedded = self.embedding(x)  # (batch, seq_len, embed_dim)
        rnn_out, hidden = self.rnn(embedded)  # (batch, seq_len, hidden_dim)
        summed = rnn_out.sum(dim=1)  # Sum over sequence length
        combined = torch.cat([summed, prev_hidden], dim=1)
        out = self.fc2(self.fc1(combined))
        return out, hidden[-1]  # (batch, 32)

class WordEnvFeaturesExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim, n_tokens):
        super().__init__(observation_space, features_dim)
        self.n_tokens = n_tokens
        self.rnn = WordRNN(n_tokens=self.n_tokens)
        # self.model = WordTransformer(n_tokens=self.n_tokens)

    def forward(self, obs):
        word = obs["word"]
        prev_hidden = obs["prev_hidden"].squeeze()
        z, h = self.rnn(word.int(), prev_hidden)

        out = torch.cat([z, h], dim=1)

        return out

