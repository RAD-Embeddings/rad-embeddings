import jax
import jraph
import distrax
import numpy as np
import jax.numpy as jnp
import flax.linen as nn
from ppo import make_train
from dfax import batch2graph
from dfa_gym import DFABisimEnv
from wrappers import LogWrapper
from flax.linen.initializers import constant, orthogonal


class GATv2Conv(nn.Module):
    out_dim: int
    num_heads: int

    def setup(self):
        self.W_s = nn.Dense(self.num_heads * self.out_dim, use_bias=False)
        self.W_t = nn.Dense(self.num_heads * self.out_dim, use_bias=False)
        self.W_e = nn.Dense(self.num_heads * self.out_dim, use_bias=False)
        self.a = nn.Dense(1, use_bias=False)

    def __call__(self, node_features: jnp.ndarray, edge_features: jnp.ndarray, edge_index: jnp.ndarray) -> jnp.ndarray:
        n_nodes = node_features.shape[0]

        src, tgt = edge_index
        src_features = node_features[src]
        tgt_features = node_features[tgt]

        h_s = self.W_s(src_features).reshape(-1, self.num_heads, self.out_dim)
        h_t = self.W_t(tgt_features).reshape(-1, self.num_heads, self.out_dim)
        h_e = self.W_e(edge_features).reshape(-1, self.num_heads, self.out_dim)

        logits = self.a(nn.leaky_relu(h_s + h_t + h_e, negative_slope=0.2))
        attn = jraph.segment_softmax(logits, src, num_segments=n_nodes)
        msgs = attn * (h_t + h_e)
        h = jraph.segment_sum(msgs, src, num_segments=n_nodes)

        return h


class Model(nn.Module):
    input_dim: int
    output_dim: int
    hidden_dim: int = 64
    num_layers: int = 8
    n_heads: int = 4

    def setup(self):
        self.linear_h = nn.Dense(self.hidden_dim)
        self.linear_e = nn.Dense(self.hidden_dim)
        self.gatv2 = GATv2Conv(out_dim=self.hidden_dim, num_heads=self.n_heads)
        self.g_embed = nn.Dense(self.output_dim)

    def __call__(
        self,
        graph
    ) -> jnp.ndarray:

        h0 = self.linear_h(graph["node_features"].astype(jnp.float32))
        e = self.linear_e(graph["edge_features"].astype(jnp.float32))
        h = h0

        mask = graph["n_states"]

        for _ in range(10):
            # h = nn.tanh(self.gatv2(jnp.concatenate([h, h0], axis=-1), e, graph["edge_index"]).sum(axis=1))
            _h = nn.tanh(self.gatv2(jnp.concatenate([h, h0], axis=-1), e, graph["edge_index"]).sum(axis=1))
            h = jnp.where((mask > 0)[:, None], _h, h)
            mask -= 1

        return self.g_embed(h[graph["current_state"]])


class ActorCritic(nn.Module):
    action_dim: int

    def setup(self):
        self.model = Model(input_dim=3, output_dim=32)
        self.value_head = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))
        self.policy_head = nn.Dense(self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0))

    def __call__(self, batch):
        
        graph_l = batch2graph(batch["graph_l"])
        graph_r = batch2graph(batch["graph_r"])

        batch = {
            "node_features": jnp.stack(jnp.array([graph_l["node_features"], graph_r["node_features"]])),
            "edge_features": jnp.stack(jnp.array([graph_l["edge_features"], graph_r["edge_features"]])),
            "edge_index": jnp.stack(jnp.array([graph_l["edge_index"], graph_r["edge_index"]])),
            "current_state": jnp.concatenate(jnp.array([graph_l["current_state"], graph_r["current_state"]])),
            "n_states": jnp.stack(jnp.array([graph_l["n_states"], graph_r["n_states"]]))
        }

        graph = batch2graph(batch)

        feat = self.model(graph)
        feat_l, feat_r = jnp.array_split(feat, 2)
        feat = jnp.concatenate([feat_l, feat_r], axis=-1)

        logits = self.policy_head(feat)
        value = self.value_head(feat)

        pi = distrax.Categorical(logits=logits)
        return pi, jnp.squeeze(value, axis=-1)


def _batchify(obss: dict, agents):
    return obss[agents[0]]

def _unbatchify(actions: jnp.ndarray, agents, n_envs):
    return {agents[0]: actions}

if __name__ == "__main__":
    config = {
        "LR": 1e-3,
        "NUM_ENVS": 8,
        "NUM_STEPS": 512,
        "TOTAL_TIMESTEPS": 1e6,
        "UPDATE_EPOCHS": 2,
        "NUM_MINIBATCHES": 4,
        "GAMMA": 0.9,
        "GAE_LAMBDA": 0.0,
        "CLIP_EPS": 0.1,
        "ENT_COEF": 0.00,
        "VF_COEF": 1.0,
        "MAX_GRAD_NORM": 0.5,
        "ANNEAL_LR": False,
        "DEBUG": True,
    }
    env = DFABisimEnv()
    env = LogWrapper(env=env, config=config)
    rng = jax.random.PRNGKey(30)
    network = ActorCritic(env.action_space(env.agents[0]).n)
    train_jit = jax.jit(make_train(config, env, network, _batchify, _unbatchify))
    out = train_jit(rng)

