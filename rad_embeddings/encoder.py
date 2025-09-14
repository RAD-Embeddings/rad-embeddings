import os
import re
import jax
import jraph
import distrax
import jax.numpy as jnp
import flax.linen as nn
from dfax import batch2graph
from dfax.samplers import RADSampler
import flax.serialization as serialization
from flax.linen.initializers import constant, orthogonal


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


class EncoderModule(nn.Module):
    encoder_dim: int = 32
    max_size: int = 10
    n_heads: int = 4

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


def distance(feat_l, feat_r):
    safe_l2_norm = lambda x: jnp.sqrt(jnp.sum(x ** 2, axis=-1, keepdims=True) + jnp.finfo(jnp.float32).eps)
    feat_l_normalized = feat_l / safe_l2_norm(feat_l)
    feat_r_normalized = feat_r / safe_l2_norm(feat_r)
    return safe_l2_norm(feat_l_normalized - feat_r_normalized)


class ActorCritic(nn.Module):
    action_dim: int
    encoder: nn.Module
    deterministic: bool = False

    def setup(self):
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

        feat = self.encoder(graph)
        feat_l, feat_r = jnp.array_split(feat, 2)

        value = distance(feat_l, feat_r)
        logits = self.policy_head(feat_l - feat_r)

        if self.deterministic:
            action = jnp.argmax(logits, axis=-1)
            return action, jnp.squeeze(value, axis=-1)
        else:
            pi = distrax.Categorical(logits=logits)
            return pi, jnp.squeeze(value, axis=-1)


class Encoder:
    def __init__(
        self,
        max_size: int = 10,
        n_tokens: int = 10,
        seed: int = 42,
    ):
        key = jax.random.PRNGKey(seed)
        self.encoder = EncoderModule(max_size=max_size)
        sampler = RADSampler()
        dfa = sampler.sample(key)
        dfa_graph = dfa.to_graph()
        self.encoder_ac = ActorCritic(action_dim=n_tokens, encoder=self.encoder, deterministic=True)
        params = self.encoder_ac.init(key, {"graph_l": dfa_graph, "graph_r": dfa_graph})

        storage_dir = os.path.join(os.path.dirname(__file__), "storage")
        pattern = re.compile(
            r"encoder_params_max_size_(\d+)_n_tokens_(\d+)_seed_(\d+)\.msgpack"
        )

        candidates = []
        for fname in os.listdir(storage_dir):
            m = pattern.match(fname)
            if m:
                f_max_size, f_n_tokens, f_seed = map(int, m.groups())
                if f_n_tokens == n_tokens and f_max_size >= max_size:
                    candidates.append((f_max_size, f_seed, fname))

        if not candidates:
            raise FileNotFoundError(
                f"No pretrained encoder found with max_size >= {max_size} "
                f"and n_tokens == {n_tokens}"
            )

        candidates.sort(key=lambda x: x[0])
        chosen = None

        for c in candidates:
            if c[1] == seed:
                chosen = c
                break
        if chosen is None:
            for c in candidates:
                if c[1] == 42:
                    chosen = c
                    break

        if chosen is None:
            raise FileNotFoundError(
                f"No pretrained encoder found for seed {seed} or fallback seed 42 "
                f"with constraints max_size >= {max_size}, n_tokens == {n_tokens}"
            )

        params_file = os.path.join(storage_dir, chosen[2])
        with open(params_file, "rb") as f:
            self.encoder_ac_params = serialization.from_bytes(params, f.read())

        self.encoder_params = {"params": self.encoder_ac_params["params"]["encoder"]}
        safe_l2_norm = lambda x: jnp.sqrt(jnp.sum(x ** 2, axis=-1, keepdims=True) + jnp.finfo(jnp.float32).eps)
        self.distance = lambda feat_l, feat_r: safe_l2_norm(
            (feat_l / safe_l2_norm(feat_l)) - (feat_r / safe_l2_norm(feat_r))
        )

    def __call__(self, graph):
        return jax.lax.stop_gradient(self.encoder.apply(self.encoder_params, graph))

    def solve(self, problem):
        action, _ = jax.lax.stop_gradient(self.encoder_ac.apply(self.encoder_ac_params, problem))
        return action

    def distance(self, feat_l, feat_r):
        return distance(feat_l, feat_r)

