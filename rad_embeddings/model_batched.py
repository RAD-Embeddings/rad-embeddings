import jax
import jax.numpy as jnp
from flax import linen as nn
import jraph
from dfax.samplers import RADSampler

class GATv2Conv(nn.Module):
    out_dim: int
    num_heads: int

    @nn.compact
    def __call__(self, node_features: jnp.ndarray, edge_features: jnp.ndarray, edge_index: jnp.ndarray) -> jnp.ndarray:
        N = node_features.shape[0]
        head_dim = self.out_dim
        W_s = nn.Dense(self.num_heads * head_dim, use_bias=False, name='W_s')
        W_t = nn.Dense(self.num_heads * head_dim, use_bias=False, name='W_t')
        W_e = nn.Dense(self.num_heads * head_dim, use_bias=False, name='W_e')
        a = nn.Dense(1, use_bias=False, name='a')

        src, tgt = edge_index
        x_s = node_features[src]
        x_t = node_features[tgt]

        h_s = W_s(x_s).reshape(-1, self.num_heads, head_dim)
        h_t = W_t(x_t).reshape(-1, self.num_heads, head_dim)
        h_e = W_e(edge_features).reshape(-1, self.num_heads, head_dim)

        logits = a(nn.leaky_relu(h_s + h_t + h_e, negative_slope=0.2))
        attn = jraph.segment_softmax(logits, src, num_segments=N)
        msgs = attn * (h_t + h_e)
        out = jraph.segment_sum(msgs, src, num_segments=N)

        return out


class Model(nn.Module):
    input_dim: int
    output_dim: int
    hidden_dim: int = 64
    num_layers: int = 8
    n_heads: int = 4

    @nn.compact
    def __call__(
        self,
        node_features: jnp.ndarray,
        edge_features: jnp.ndarray,
        edge_index: jnp.ndarray,
        current_state: jnp.ndarray,
        n_reach_states: int
    ) -> jnp.ndarray:

        linear_in = nn.Dense(self.hidden_dim, name='linear_in')
        conv = GATv2Conv(out_dim=self.hidden_dim, num_heads=self.n_heads)
        activation = jnp.tanh
        g_embed = nn.Dense(self.output_dim, name='g_embed')

        h0 = linear_in(node_features.astype(jnp.float32))  # [N, hidden_dim]
        h = h0

        for _ in range(10):
            h = conv(jnp.concatenate([h, h0], axis=-1), edge_features, edge_index).sum(axis=1)
            h = activation(h)

        return g_embed(h[current_state])


if __name__ == "__main__":

    key = jax.random.PRNGKey(0)
    _dfax = RADSampler().sample(key)
    node_features, edge_features, edge_index, current_state = _dfax.to_graph()
    n_reach_states = _dfax.n_states()
    print(node_features)
    print(edge_features)
    print(edge_index)
    print(current_state)
    print(n_reach_states)


    # Initialize and run
    F_in = 3  # input feature dimension
    F_out = 32  # output feature dimension
    model = Model(input_dim=F_in, output_dim=F_out)
    params = model.init(key, node_features, edge_features, edge_index, current_state, n_reach_states)

    # @jax.jit
    # def run_apply(params, node_features, edge_features, edge_index, current_state, n_reach_states):
    #     return model.apply(params, node_features, edge_features, edge_index, current_state, n_reach_states)

    # preds = run_apply(params, node_features, edge_features, edge_index, current_state, n_reach_states)
    preds = model.apply(params, node_features, edge_features, edge_index, current_state, n_reach_states)

    print("Output shape:", preds.shape)
    print("Predictions:\n", preds)

