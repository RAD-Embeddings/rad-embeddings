import jax
import jraph
import jax.numpy as jnp
import flax.linen as nn
from dfax.samplers import RADSampler
import flax.serialization as serialization


class GATv2Conv(nn.Module):
    out_dim: int
    num_heads: int

    def setup(self):
        self.W_s = nn.Dense(self.num_heads * self.out_dim, use_bias=False)
        self.W_et = nn.Dense(self.num_heads * self.out_dim, use_bias=False)
        self.a = nn.Dense(1, use_bias=False)

    def __call__(self, node_features: jnp.ndarray, edge_features: jnp.ndarray, edge_index: jnp.ndarray) -> jnp.ndarray:
        n_nodes = node_features.shape[0]

        src, tgt = edge_index
        src_features = node_features[src]
        tgt_features = node_features[tgt]
        edge_tgt_features = jnp.concatenate([tgt_features, edge_features], axis=-1)

        h_s = self.W_s(src_features).reshape(-1, self.num_heads, self.out_dim)
        h_et = self.W_et(edge_tgt_features).reshape(-1, self.num_heads, self.out_dim)

        logits = self.a(nn.leaky_relu(h_s + h_et, negative_slope=0.2))
        attn = jraph.segment_softmax(logits, src, num_segments=n_nodes)
        msgs = attn * h_et
        h = jraph.segment_sum(msgs, src, num_segments=n_nodes)

        return h


class Encoder(nn.Module):
    output_dim: int
    hidden_dim: int = 64
    n_msg_stps: int = 10
    n_heads: int = 4

    @staticmethod
    def load_params(output_dim, n_msg_stps, encoder_dir):
        encoder = Encoder(output_dim=output_dim, n_msg_stps=n_msg_stps)
        sampler = RADSampler(p=None)
        rng = jax.random.PRNGKey(30)
        dfa = sampler.sample(rng)
        dfa_graph = dfa.to_graph()
        network_params = encoder.init(rng, dfa_graph)
        with open(encoder_dir, "rb") as f:
            encoder_params = serialization.from_bytes(network_params, f.read())
        return encoder, encoder_params

    def setup(self):
        self.linear_h = nn.Dense(self.hidden_dim)
        self.linear_e = nn.Dense(self.hidden_dim)
        self.gatv2 = GATv2Conv(out_dim=self.hidden_dim, num_heads=self.n_heads)
        self.g_embed = nn.Dense(self.output_dim)

    def __call__(
        self,
        graph
    ) -> jnp.ndarray:

        node_mask = jnp.any(graph["node_features"] != 0, axis=-1)
        h0 = self.linear_h(graph["node_features"].astype(jnp.float32))
        h0 = jnp.where(node_mask[:, None], h0, 0)

        edge_mask = jnp.any(graph["edge_features"] != 0, axis=-1)
        e = self.linear_e(graph["edge_features"].astype(jnp.float32))
        e = jnp.where(edge_mask[:, None], e, 0)

        h = h0

        mask = graph["n_states"]

        for _ in range(self.n_msg_stps):
            # h = nn.tanh(self.gatv2(jnp.concatenate([h, h0], axis=-1), e, graph["edge_index"]).sum(axis=1))
            _h = nn.tanh(self.gatv2(jnp.concatenate([h, h0], axis=-1), e, graph["edge_index"]).sum(axis=1))
            h = jnp.where((mask > 0)[:, None], _h, h)
            mask -= 1

        return self.g_embed(h[graph["current_state"]])

