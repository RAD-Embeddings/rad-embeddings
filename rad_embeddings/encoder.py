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
        # self.W_s = nn.Dense(self.num_heads * self.out_dim, use_bias=False)
        # self.W_t = nn.Dense(self.num_heads * self.out_dim, use_bias=False)
        # self.W_e = nn.Dense(self.num_heads * self.out_dim, use_bias=False)
        self.W = nn.Dense(self.num_heads * self.out_dim, use_bias=False)
        self.a = nn.Dense(1, use_bias=False)

    def __call__(self, node_features: jnp.ndarray, edge_features: jnp.ndarray, edge_index: jnp.ndarray) -> jnp.ndarray:
        n_nodes = node_features.shape[0]

        src, tgt = edge_index
        src_features = node_features[src]
        tgt_features = node_features[tgt]

        mask = jnp.any(edge_features != 0, axis=-1)

        z = jnp.concatenate([src_features, edge_features, tgt_features], axis=-1)
        h = self.W(z).reshape(-1, self.num_heads, self.out_dim)

        # h_s = self.W_s(src_features).reshape(-1, self.num_heads, self.out_dim)
        # h_s = jnp.where(mask[:, None, None], h_s, 0)

        # h_t = self.W_t(tgt_features).reshape(-1, self.num_heads, self.out_dim)
        # h_t = jnp.where(mask[:, None, None], h_t, 0)

        # h_e = self.W_e(edge_features).reshape(-1, self.num_heads, self.out_dim)
        # h_e = jnp.where(mask[:, None, None], h_e, 0)

        # logits = self.a(nn.leaky_relu(h_s + h_t + h_e, negative_slope=0.2))
        logits = self.a(nn.leaky_relu(h, negative_slope=0.2))
        logits = jnp.where(
            mask[:, None, None] > 0,
            logits,
            -jnp.inf
        )
        max_per_node = jraph.segment_max(logits.reshape(logits.shape[0], -1),
                                         src,
                                         num_segments=n_nodes)
        dead_nodes = jnp.isneginf(max_per_node[:, 0])
        dead_mask = dead_nodes[src]
        safe_logits = jnp.where(dead_mask[:, None, None], 0.0, logits)
        attn = jraph.segment_softmax(safe_logits, src, n_nodes)
        msgs = attn * h
        h = jraph.segment_sum(msgs, src, num_segments=n_nodes)

        return h


class Encoder(nn.Module):
    output_dim: int
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
        hidden_dim = self.output_dim * 2
        self.linear_h = nn.Dense(hidden_dim)
        self.linear_e = nn.Dense(hidden_dim)
        self.gatv2 = GATv2Conv(out_dim=hidden_dim, num_heads=self.n_heads)
        self.g_embed = nn.Dense(self.output_dim)

    def __call__(
        self,
        graph
    ) -> jnp.ndarray:

        h0 = self.linear_h(graph["node_features"].astype(jnp.float32))
        h0 = jnp.where(jnp.any(graph["node_features"] != 0, axis=-1)[:, None], h0, 0)
        e = self.linear_e(graph["edge_features"].astype(jnp.float32))

        h = h0

        mask = graph["n_states"]

        for _ in range(self.n_msg_stps):
            # h = nn.tanh(self.gatv2(jnp.concatenate([h, h0], axis=-1), e, graph["edge_index"]).sum(axis=1))
            _h = nn.tanh(self.gatv2(jnp.concatenate([h, h0], axis=-1), e, graph["edge_index"]).sum(axis=1))
            h = jnp.where((mask > 0)[:, None], _h, h)
            mask -= 1

        return self.g_embed(h[graph["current_state"]])

