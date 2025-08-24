import os
import jax
import wandb
import jraph
import distrax
import argparse
import numpy as np
import jax.numpy as jnp
import flax.linen as nn
from ppo import make_train
from encoder import Encoder
from dfax import batch2graph
from wrappers import LogWrapper
from flax.core import FrozenDict
from dfa_gym import TokenEnv, DFAWrapper
import flax.serialization as serialization
from dfax.samplers import ReachAvoidSampler
from flax.linen.initializers import constant, orthogonal


class CNN(nn.Module):
    dims: list[int]

    @nn.compact
    def __call__(self, x):
        for dim in self.dims:
            x = nn.Conv(dim, (2, 2), strides=(1, 1), kernel_init=orthogonal(np.sqrt(2)))(x)
            x = nn.relu(x)
        return x.reshape((x.shape[0], -1))


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
    encoder: nn.Module
    encoder_params: FrozenDict
    freeze_encoder: bool = True

    def setup(self):
        self.cnn = CNN([16, 32, 64])
        self.value_feat = MLP([64, 64])
        self.policy_feat = MLP([64, 64, 64])
        self.value_net = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))
        self.policy_net = nn.Dense(self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0))

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

        dfa_feat = jnp.where(self.freeze_encoder,
            jax.lax.stop_gradient(self.encoder.apply(self.encoder_params, graph)),
            self.encoder.apply(self.encoder_params, graph))

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
        "LR": 3e-4,
        "NUM_ENVS": 16,
        "NUM_STEPS": 128,
        "TOTAL_TIMESTEPS": 1e7,
        "UPDATE_EPOCHS": 10,
        "NUM_MINIBATCHES": 8,
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
    parser.add_argument(
        "--rad-dim",
        type=int,
        default=32,
        help="Size of the RAD embeddings"
    )
    parser.add_argument(
        "--freeze-encoder",
        type=bool,
        default=True,
        help="Freeze the encoder"
    )
    parser.add_argument(
        "--wandb",
        action="store_true",
        help="Log to wandb"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Print logs"
    )
    args = parser.parse_args()

    config["DEBUG"] = args.debug
    config["WANDB"] = args.wandb

    if config["WANDB"]:
        wandb.init(
            entity="beyazit-y-berkeley-eecs",
            project="rad-marl-jax",
            config=config
        )

    rng = jax.random.PRNGKey(args.seed)

    env = DFAWrapper(TokenEnv(), sampler=ReachAvoidSampler(max_size=6))
    env = LogWrapper(env=env, config=config)

    encoder, encoder_params = Encoder.load_params(
        output_dim=args.rad_dim,
        n_msg_stps=env.sampler.max_size,
        encoder_dir=f"{args.save_dir}/trained_encoder_params_{args.seed}.msgpack"
    )

    network = ActorCritic(
        action_dim=env.action_space(env.agents[0]).n,
        encoder=encoder,
        encoder_params=encoder_params,
        freeze_encoder=args.freeze_encoder
    )

    train_jit = jax.jit(make_train(config, env, network, _batchify))
    out = train_jit(rng)

    os.makedirs(args.save_dir, exist_ok=True)
    trained_params = out["runner_state"][0].params
    with open(f"{args.save_dir}/trained_token_env_policy_params_{args.seed}.msgpack", "wb") as f:
        f.write(serialization.to_bytes(trained_params))

    if config["WANDB"]:
        wandb.finish()

