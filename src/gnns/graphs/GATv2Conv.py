import numpy as np
import torch
import torch.nn as nn

import dgl
from dgl.nn.pytorch.conv import GATv2Conv

from gnns.graphs.GNN import GNN

class GATv2ConvEncoder(GNN):
    def __init__(self, input_dim, output_dim, **kwargs):
        super().__init__(input_dim, output_dim)

        from utils.env import edge_types

        hidden_dim = kwargs.get('hidden_dim', 32)
        num_layers = kwargs.get('num_layers', 8)
        n_heads = kwargs.get('n_heads', 4)

        self.num_layers = num_layers

        self.linear_in = nn.Linear(input_dim, (n_heads//2)*hidden_dim)
        self.conv = GATv2Conv(n_heads*hidden_dim, (n_heads//2)*hidden_dim, n_heads, activation=torch.tanh)
        self.g_embed = nn.Linear((n_heads//2)*hidden_dim, output_dim)

    def forward(self, g):
        g = np.array(g).reshape((1, -1)).tolist()[0]
        g = dgl.batch(g)
        h_0 = self.linear_in(g.ndata["feat"].float().squeeze(dim=1))
        h = h_0
        for i in range(self.num_layers):
            h = self.conv(g, torch.cat([h, h_0], dim=1)).sum(dim=1)
        g.ndata['h'] = h
        g.ndata["is_root"] = g.ndata["is_root"].float()
        hg = dgl.sum_nodes(g, 'h', weight='is_root')
        return self.g_embed(hg)
