import jax
import jraph
import distrax
import argparse
import numpy as np
import jax.numpy as jnp
import flax.linen as nn
from ppo import make_train
from encoder import Encoder
from dfax import batch2graph
from dfa_gym import TokenEnv, DFAWrapper
from wrappers import LogWrapper
from flax.linen.initializers import constant, orthogonal


class CNN(nn.Module):

    @nn.compact
    def __call__(self, x):
        x = nn.Conv(16, (2, 2), strides=(1, 1), kernel_init=orthogonal(np.sqrt(2)))(x)
        x = nn.relu(x)
        x = nn.Conv(32, (2, 2), strides=(1, 1), kernel_init=orthogonal(np.sqrt(2)))(x)
        x = nn.relu(x)
        x = nn.Conv(64, (2, 2), strides=(1, 1), kernel_init=orthogonal(np.sqrt(2)))(x)
        x = nn.relu(x)
        return x.reshape((x.shape[0], -1)) # Flatten (start_dim=B)


class MLP(nn.Module):
    dims: list[int]

    @nn.compact
    def __call__(self, x):
        for dim in self.dims:
            x = nn.Dense(dim, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(x)
            x = nn.relu(x)
        return x


class ActorCritic(nn.Module):
    action_dim: int
    encoder_dir: str

    def setup(self):
        self.cnn = CNN()
        self.encoder = Encoder(output_dim=32, num_layers=10)
        self.value_feat = MLP([64, 64])
        self.policy_feat = MLP([64, 64, 64])
        self.value_net = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))
        self.policy_net = nn.Dense(self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0))
        self.pretrained_encoder_params = Encoder.load_params(output_dim=32, n_msg_stps=10, encoder_dir=self.encoder_dir)

    @nn.compact
    def __call__(self, wrapped_obs):
        obs = wrapped_obs["obs"]
        graph = wrapped_obs["graph"]

        if obs.ndim == 3: # (C, H, W)
            obs = obs[None, ...] # -> (1, C, H, W)
        elif obs.ndim != 4:
            raise ValueError(f"Expected (C, H, W) or (B, C, H, W), got {obs.shape}")

        obs = jnp.transpose(obs, (0, 2, 3, 1)) # -> (B, H, W, C)
        obs_feat = self.cnn(obs)

        graph = batch2graph(graph)

        dfa_feat = jax.lax.stop_gradient(self.encoder.apply(self.pretrained_encoder_params, graph))

        feat = jnp.concatenate([obs_feat, dfa_feat], axis=-1)

        value_hidden = self.value_feat(feat)
        policy_hidden = self.policy_feat(feat)

        value = self.value_net(value_hidden)
        logits = self.policy_net(policy_hidden)

        pi = distrax.Categorical(logits=logits)
        return pi, jnp.squeeze(value, axis=-1)


def _batchify(obss: dict, agents):

    obs = jnp.concatenate([obss[agent]["obs"] for agent in agents])

    node_features_batch = jnp.concatenate([obss[agent]["graph"]["node_features"] for agent in agents])
    edge_features_batch = jnp.concatenate([obss[agent]["graph"]["edge_features"] for agent in agents])
    edge_index_batch = jnp.concatenate([obss[agent]["graph"]["edge_index"] for agent in agents])
    current_state_batch = jnp.concatenate(jnp.array([obss[agent]["graph"]["current_state"] for agent in agents]))
    n_states_batch = jnp.concatenate(jnp.array([obss[agent]["graph"]["n_states"] for agent in agents]))

    batch = {
        "node_features": node_features_batch,
        "edge_features": edge_features_batch,
        "edge_index": edge_index_batch,
        "current_state": current_state_batch,
        "n_states": n_states_batch,
    }

    return {"obs": obs, "graph": batch}


if __name__ == "__main__":
    config = {
        "LR": 2.5e-4,
        "NUM_ENVS": 16,
        "NUM_STEPS": 128,
        "TOTAL_TIMESTEPS": 1e6,
        "UPDATE_EPOCHS": 10,
        "NUM_MINIBATCHES": 4,
        "GAMMA": 0.99,
        "GAE_LAMBDA": 0.95,
        "CLIP_EPS": 0.2,
        "ENT_COEF": 0.0,
        "VF_COEF": 0.5,
        "MAX_GRAD_NORM": 0.5,
        "ANNEAL_LR": False,
        "DEBUG": True,
    }

    parser = argparse.ArgumentParser(description="Train DFA encoder")
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed used for PRNGKey"
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default="storage",
        help="Directory for saving the trained encoder"
    )
    args = parser.parse_args()

    rng = jax.random.PRNGKey(args.seed)

    env = DFAWrapper(TokenEnv())
    env = LogWrapper(env=env, config=config)

    network = ActorCritic(
        action_dim=env.action_space(env.agents[0]).n,
        encoder_dir=f"{args.save_dir}/trained_encoder_params_{args.seed}.msgpack"
    )

    train_jit = jax.jit(make_train(config, env, network, _batchify))
    out = train_jit(rng)

