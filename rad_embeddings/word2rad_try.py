import torch
import torch.nn as nn
import torch.optim as optim

from dfa_samplers import RADSampler
from dfa.utils import paths, words

import torch.nn.functional as F

from encoder import Encoder

import numpy as np

import threading

from dfa.utils import words, paths, dfa2dict, min_distance_to_accept_by_state

from torch.utils.tensorboard import SummaryWriter


# def pad_word(w, w_length):
#     return F.pad(input=torch.tensor(w) + 1, pad=(0, w_length - len(w)), mode='constant', value=0).int()  # 0 is the empty string


# def generate_problem(batch_size, w_length):
#     sampler = RADSampler()
#     dfa = None
#     words = None
#     while dfa == None and words == None:
#         try:
#             dfa = sampler.sample()
#             gen_word = paths(dfa, dfa.start, end=dfa.advance(dfa.find_word()).start, max_length=w_length, randomize=False)
#             words = torch.stack([pad_word(next(gen_word), w_length) for w in range(batch_size)])
#         except:
#             dfa = None
#             words = None
#     return dfa, words

def pad_word(w, w_length):
    return torch.from_numpy(np.pad(np.array(w, dtype=np.int32) + 1, pad_width=(0, w_length - len(w)), mode='constant', constant_values=0)) # 0 is the empty string

def softmax(x, axis=1):
    x_max = np.max(x, axis=axis).reshape(-1, 1)
    exp_x = np.exp(x - x_max)
    x_sum = np.sum(exp_x, axis=axis).reshape(-1, 1)
    return exp_x / x_sum

def problem_generator(sampler, w_length, n_samples):
    result = [None]
    def _word_generator():
        dfa = sampler.sample()
        dfa_dict = dfa2dict(dfa)[0]
        dists = np.array([[min_distance_to_accept_by_state(dfa)[j] for j in list(dfa_dict[i][1].values())] for i in dfa_dict.keys()], dtype=np.float32)
        prob = softmax(-dists)
        words = []
        while len(words) < n_samples:
            word = []
            s = 0
            n_tokens = prob.shape[1]
            label = False
            for _ in range(w_length):
                a = np.random.choice(n_tokens, p=prob[s])
                label, transitions = dfa_dict[s]
                s = transitions[a]
                word.append(a)
                if label:
                    break
            if label:
                words.append(pad_word(word, w_length))
        assert len(words) == n_samples
        result[0] = (dfa, torch.from_numpy(np.array(words)))
    while True:
        thread = threading.Thread(target=_word_generator)
        thread.start()
        thread.join(2) # seconds

        if thread.is_alive():
            thread.join()  # Ensure thread is killed before retrying
        else:
            yield result[0]

    

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
        words = self.fc(summed)
        rad = words.sum(dim=0).unsqueeze(dim=0)
        return rad  # (batch, 32)


class WordTransformer(nn.Module):
    def __init__(self, n_tokens, embed_dim, num_heads, hidden_dim, num_layers, output_dim, w_length):
        super().__init__()
        self.embedding = nn.Embedding(n_tokens + 1, embed_dim)
        self.pos_encoding = nn.Parameter(torch.randn(1, w_length, embed_dim))  # Max sequence length = w_length
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(embed_dim, num_heads, hidden_dim, batch_first=True),
            num_layers
        )
        self.fc = nn.Linear(embed_dim, output_dim)  # Project to DFA embedding space

    def forward(self, x):
        x = self.embedding(x) + self.pos_encoding[:, :x.shape[1], :]
        x = self.transformer(x)
        x = x.mean(dim=1)  # Pool over sequence dimension
        words = self.fc(x)
        rad = words.mean(dim=0).unsqueeze(dim=0)
        return rad

# Hyperparameters
n_tokens = 10  # Adjust as needed
embed_dim = 64
num_heads = 2
hidden_dim = 64
num_layers = 2
output_dim = 32

learning_rate = 0.001
num_epochs = 1_000
batch_size = 64

w_length = 10

model_type = "transformer_2_heads_2_layers"

# Model, loss, and optimizer
model = StringRNN(n_tokens, embed_dim, hidden_dim, output_dim) if model_type == "rnn" else WordTransformer(n_tokens, embed_dim, num_heads, hidden_dim, num_layers, output_dim, w_length)

print("Total number of parameters:", sum(p.numel() for p in model.parameters() if p.requires_grad))
print(model)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

encoder = Encoder(load_file="storage/DFABisimEnv-v1-encoder")

gen_problem = problem_generator(sampler=RADSampler(), w_length=w_length, n_samples=100)


writer = SummaryWriter(log_dir=f"storage/runs/word2rad/{model_type}")

# Training loop
for epoch in range(num_epochs):
    batch_losses = []
    for _ in range(batch_size):
        dfa, words = next(gen_problem)
        rad = encoder.dfa2rad(dfa)
        # embed = torch.sum(model(words), dim=0).unsqueeze(dim=0)
        radish = model(words)
        # from torchviz import make_dot
        # make_dot(radish).save()
        batch_losses.append(encoder.rad2val(torch.cat([rad, radish], dim=1)))
    loss = torch.stack(batch_losses).mean()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    writer.add_scalar('Loss/train', loss.item(), epoch)

    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

print("Training complete!")
writer.close()

torch.save(model.state_dict(), f"storage/runs/word2rad/{model_type}/model")
model.load_state_dict(torch.load(f"storage/runs/word2rad/{model_type}/model", weights_only=True))
model.eval()
