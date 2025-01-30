import torch
import torch.nn as nn
import torch.optim as optim

class WordTransformer(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_heads, hidden_dim, num_layers, dfa_embed_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_encoding = nn.Parameter(torch.randn(1, 100, embed_dim))  # Max sequence length = 100
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(embed_dim, num_heads, hidden_dim, batch_first=True),
            num_layers
        )
        self.fc = nn.Linear(embed_dim, dfa_embed_dim)  # Project to DFA embedding space

    def forward(self, x):
        x = self.embedding(x) + self.pos_encoding[:, :x.shape[1], :]
        x = self.transformer(x)
        x = x.mean(dim=1)  # Pool over sequence dimension
        return self.fc(x)  # Output DFA embedding

# Model hyperparameters
vocab_size = 5000  # Adjust based on dataset
embed_dim = 128
num_heads = 4
hidden_dim = 256
num_layers = 3
dfa_embed_dim = 64  # Dimension of \phi(A)

model = DFATransformer(vocab_size, embed_dim, num_heads, hidden_dim, num_layers, dfa_embed_dim)

# Example input: sequence of word indices
example_input = torch.randint(0, vocab_size, (32, 20))  # (batch_size=32, sequence_length=20)
output = model(example_input)  # Output shape: (32, dfa_embed_dim)
