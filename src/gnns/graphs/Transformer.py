import torch
import torch.nn as nn
import torch.nn.functional as F

import dgl
import numpy as np

class TranformerEncoder(nn.Module):
    def __init__(self, input_dim, output_dim, **kwargs):
        super().__init__()

        hidden_dim = kwargs.get('hidden_dim', 32)
        depth = kwargs.get('depth', 8)
        n_heads = kwargs.get('n_heads', 2)
        pos_enc_size = kwargs.get('pos_enc_size', 2)

        self.pos_enc_size = pos_enc_size
        self.pre = nn.Linear(input_dim, hidden_dim)
        self.pre_pos = nn.Linear(self.pos_enc_size, hidden_dim)
        self.transformer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=n_heads, dim_feedforward=256)
        self.post = nn.Linear(hidden_dim, output_dim)

        self.depth = depth

    def forward(self, g):
        g = np.array(g).reshape((1, -1)).tolist()[0]
        g = dgl.batch(g)
        h = g.ndata["feat"].float().squeeze(dim=1)

        h = self.pre(h) + self.pre_pos(g.ndata["PE"])
        
        depth_hop_matrix = dgl.khop_adj(g, self.depth)
        # depth_hop_matrix = dgl.khop_adj(g, self.depth).transpose(0, 1)
        depth_hop_matrix = depth_hop_matrix.masked_fill(depth_hop_matrix == 0, float("inf"))

        mask = 1 - depth_hop_matrix

        h = self.transformer(h, mask)

        g.ndata['h'] = h
        g.ndata["is_root"] = g.ndata["is_root"].float()
        hg = dgl.sum_nodes(g, 'h', weight='is_root')

        return self.post(hg)

