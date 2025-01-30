import torch
import torch.nn as nn
import torch.optim as optim

from dfa_samplers import RADSampler
from dfa.utils import paths, words

import torch.nn.functional as F

from encoder import Encoder

def pad_word(w, w_length):
    return F.pad(input=torch.tensor(w) + 1, pad=(0, w_length - len(w)), mode='constant', value=0).int()  # 0 is the empty string


def generate_problem(batch_size, w_length):
    sampler = RADSampler()
    dfa = None
    words = None
    while dfa == None and words == None:
        try:
            dfa = sampler.sample()
            gen_word = paths(dfa, dfa.start, end=dfa.advance(dfa.find_word()).start, max_length=w_length, randomize=False)
            words = torch.stack([pad_word(next(gen_word), w_length) for w in range(batch_size)])
        except:
            dfa = None
            words = None
    return dfa, words

    

class StringRNN(nn.Module):
    def __init__(self, n_tokens, embed_dim, hidden_dim, output_dim):
        super().__init__()
        self.embedding = nn.Embedding(n_tokens + 1, embed_dim) # + 1 is for the empty string
        self.rnn = nn.RNN(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)  # Map summed RNN output to 32-dim vector

    def forward(self, x):
        embedded = self.embedding(x)  # (batch, seq_len, embed_dim)
        rnn_out, _ = self.rnn(embedded)  # (batch, seq_len, hidden_dim)
        summed = rnn_out.sum(dim=1)  # Sum over sequence length
        return self.fc(summed)  # (batch, 32)

# Hyperparameters
n_tokens = 10  # Adjust as needed
embed_dim = 64
hidden_dim = 64
output_dim = 32

learning_rate = 0.001
num_epochs = 1_000
batch_size = 10

w_length = 10


# Model, loss, and optimizer
model = StringRNN(n_tokens, embed_dim, hidden_dim, output_dim)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

encoder = Encoder(load_file="storage/DFABisimEnv-v1-encoder")

# Training loop
for epoch in range(num_epochs):
    loss = 0
    for _ in range(batch_size):
        dfa, words = generate_problem(batch_size, w_length)
        rad = encoder.dfa2rad(dfa)
        embed = torch.mean(model(words), dim=0).unsqueeze(dim=0)
        loss += encoder.rad2val(torch.cat([rad, embed], dim=1))
    optimizer.zero_grad()
    loss.mean().backward()
    optimizer.step()

    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

print("Training complete!")
