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
        self.W_a = nn.Dense(self.num_heads * self.out_dim, use_bias=False)
        self.W_m = nn.Dense(self.num_heads * self.out_dim, use_bias=False)
        self.a = nn.Dense(1, use_bias=False)

    def __call__(self, node_features: jnp.ndarray, edge_features: jnp.ndarray, edge_index: jnp.ndarray, attn_mask: jnp.ndarray) -> jnp.ndarray:
        n_nodes = node_features.shape[0]

        src, tgt = edge_index
        src_features = node_features[src]
        tgt_features = node_features[tgt]

        h_a = self.W_a(
            jnp.concatenate([src_features, edge_features, tgt_features], axis=-1)
        ).reshape(-1, self.num_heads, self.out_dim)

        h_m = self.W_m(
            jnp.concatenate([edge_features, tgt_features], axis=-1)
        ).reshape(-1, self.num_heads, self.out_dim)

        logits = self.a(nn.leaky_relu(h_a, negative_slope=0.2))
        logits = jnp.where(
            attn_mask[:, None, None],
            logits,
            -jnp.inf
        )
        max_per_node = jraph.segment_max(logits.reshape(logits.shape[0], -1),
                                         src,
                                         num_segments=n_nodes)
        dead_nodes = jnp.isneginf(max_per_node[:, 0])
        dead_mask = dead_nodes[src]
        safe_logits = jnp.where(
            dead_mask[:, None, None],
            jnp.zeros_like(logits),
            logits
        )
        attn = jraph.segment_softmax(safe_logits, src, n_nodes)
        msgs = attn * h_m
        h = jraph.segment_sum(msgs, src, num_segments=n_nodes)

        return h


class Encoder(nn.Module):
    encoder_dim: int
    max_size: int = 10
    n_heads: int = 4

    @staticmethod
    def load_params(encoder_file, encoder_dim, max_size=10, n_heads=4):
        encoder = Encoder(encoder_dim=encoder_dim, max_size=max_size, n_heads=n_heads)
        sampler = RADSampler(p=None)
        rng = jax.random.PRNGKey(30)
        dfa = sampler.sample(rng)
        dfa_graph = dfa.to_graph()
        network_params = encoder.init(rng, dfa_graph)
        with open(encoder_file, "rb") as f:
            encoder_params = serialization.from_bytes(network_params, f.read())
        return encoder, encoder_params

    def setup(self):
        hidden_dim = self.encoder_dim * 2
        self.linear_h = nn.Dense(hidden_dim, use_bias=False)
        self.linear_e = nn.Dense(hidden_dim, use_bias=False)
        self.gatv2 = GATv2Conv(out_dim=hidden_dim, num_heads=self.n_heads)
        self.g_embed = nn.Dense(self.encoder_dim, use_bias=False)

    def __call__(
        self,
        graph
    ) -> jnp.ndarray:

        h0 = self.linear_h(graph["node_features"].astype(jnp.float32))
        e = self.linear_e(graph["edge_features"].astype(jnp.float32))
        edge_index = graph["edge_index"]
        attn_mask = jnp.any(graph["edge_features"] != 0, axis=-1)

        h = h0
        n_states = graph["n_states"]

        for i in range(self.max_size):
            _h = nn.tanh(
                self.gatv2(
                    node_features=jnp.concatenate([h, h0], axis=-1),
                    edge_features=e,
                    edge_index=edge_index,
                    attn_mask=attn_mask
                ).sum(axis=1)
            )
            h = jnp.where((i < n_states)[:, None], _h, h)

        return self.g_embed(h[graph["current_state"]])

