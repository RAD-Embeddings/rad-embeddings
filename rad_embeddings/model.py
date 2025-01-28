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
        node_mask = data.node_mask

        h_0 = self.linear_in(feat.float())
        h = h_0.clone()

        # Precompute all edge indices
        edge_indices = [
            Batch.from_data_list(data.index_select(i < data.n_states)).edge_index
            for i in range(data.edge_mask.max().item())
        ]

        # Track intermediate states
        h_history = [h]

        for i, edges in enumerate(edge_indices):
            # Create persistent computation node
            mask = i < node_mask
            active_nodes = mask.nonzero().squeeze()

            # Compute updates without in-place ops
            h_next = h.clone()
            h_next[active_nodes] = self.conv(
                torch.cat([h[active_nodes], h_0[active_nodes]], 1),
                edges
            ).view(-1, self.n_heads, self.hidden_dim).sum(1)

            h_next[active_nodes] = self.activation(h_next[active_nodes])
            h = h_next
            h_history.append(h)  # Preserve gradients

        hg = h[current_state.bool()]
        return self.g_embed(hg)
