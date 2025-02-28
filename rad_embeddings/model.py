import torch
import torch.nn as nn
from torch_geometric.data import Batch
from torch_geometric.nn import GATv2Conv

class Model(nn.Module):
    def __init__(self, input_dim, output_dim, **kwargs):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = kwargs.get("hidden_dim", 64)
        self.num_layers = kwargs.get("num_layers", 8)
        self.n_heads = kwargs.get("n_heads", 4)
        self.linear_in = nn.Linear(self.input_dim, self.hidden_dim)
        self.conv = GATv2Conv(2*self.hidden_dim, self.hidden_dim, heads=self.n_heads, add_self_loops=True)
        self.activation = nn.Tanh()
        self.g_embed = nn.Linear(self.hidden_dim, self.output_dim)

    def forward(self, data):
        feat = data.feat
        edge_index = data.edge_index
        current_state = data.current_state
        h_0 = self.linear_in(feat.float())
        h = h_0
        for _ in range(8):
            h = self.conv(
                torch.cat([h, h_0], 1),
                edge_index
            ).view(-1, self.n_heads, self.hidden_dim).sum(1)
            h = self.activation(h)
        hg = h[current_state.bool()]
        return self.g_embed(hg)

    # def forward(self, data):
    #     feat = data.feat
    #     edge_index = data.edge_index
    #     current_state = data.current_state
    #     active_node_indices = data.active_node_indices
    #     active_edge_indices = data.active_edge_indices
    #     h_0 = self.linear_in(feat.float())
    #     h = h_0.clone()
    #     # Track intermediate states
    #     h_history = [h]
    #     for active_node_idx, active_edge_idx  in zip(active_node_indices, active_edge_indices):
    #         # Get active edges
    #         active_edges = edge_index[:, active_edge_idx]
    #         # Shift indices appearing in active edges
    #         edges = active_edges - torch.cumsum(~active_node_idx, dim=0)[active_edges]
    #         # Compute updates without in-place ops
    #         h_next = h.clone()
    #         h_next[active_node_idx] = self.conv(
    #             torch.cat([h[active_node_idx], h_0[active_node_idx]], 1),
    #             edges
    #         ).view(-1, self.n_heads, self.hidden_dim).sum(1)
    #         h_next[active_node_idx] = self.activation(h_next[active_node_idx])
    #         h = h_next
    #         # Preserve gradients
    #         h_history.append(h)
    #     hg = h[current_state.bool()]
    #     return self.g_embed(hg)
