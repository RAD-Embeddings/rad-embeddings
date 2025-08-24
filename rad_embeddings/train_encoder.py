import os
import jax
import wandb
import jraph
import distrax
import argparse
import jax.numpy as jnp
import flax.linen as nn
from ppo import make_train
from encoder import Encoder
from dfax import batch2graph
from dfa_gym import DFABisimEnv
from wrappers import LogWrapper
from utils import summarize_params
import flax.serialization as serialization
from flax.linen.initializers import constant, orthogonal


class ActorCritic(nn.Module):
    action_dim: int
    encoder: nn.Module

    def setup(self):
        self.safe_l2_norm = lambda x: jnp.sqrt(jnp.sum(x ** 2, axis=-1, keepdims=True) + 1e-8)
        self.policy_head = nn.Dense(self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0))

    def __call__(self, batch):
        
        graph_l = batch2graph(batch["graph_l"])
        graph_r = batch2graph(batch["graph_r"])

        batch = {
            "node_features": jnp.stack(jnp.array([graph_l["node_features"], graph_r["node_features"]])),
            # "edge_features": jnp.stack(jnp.array([graph_l["edge_features"], graph_r["edge_features"]])),
            "edge_index": jnp.stack(jnp.array([graph_l["edge_index"], graph_r["edge_index"]])),
            "current_state": jnp.concatenate(jnp.array([graph_l["current_state"], graph_r["current_state"]])),
            "n_states": jnp.stack(jnp.array([graph_l["n_states"], graph_r["n_states"]]))
        }

        graph = batch2graph(batch)

        feat = self.encoder(graph)
        feat_l, feat_r = jnp.array_split(feat, 2)

        feat_l_normalized = feat_l / self.safe_l2_norm(feat_l)
        feat_r_normalized = feat_r / self.safe_l2_norm(feat_r)
        value = self.safe_l2_norm(feat_l_normalized - feat_r_normalized)

        feat = jnp.concatenate([feat_l, feat_r], axis=-1)
        logits = self.policy_head(feat)

        pi = distrax.Categorical(logits=logits)
        return pi, value.squeeze()


def _batchify(obss: dict, agents):
    return obss[agents[0]]


if __name__ == "__main__":
    config = {
        "LR": 1e-3,
        "NUM_ENVS": 16,
        "NUM_STEPS": 512,
        "TOTAL_TIMESTEPS": 1e6,
        "UPDATE_EPOCHS": 2,
        "NUM_MINIBATCHES": 8,
        "GAMMA": 0.9,
        "GAE_LAMBDA": 0.0,
        "CLIP_EPS": 0.1,
        "ENT_COEF": 0.00,
        "VF_COEF": 1.0,
        "MAX_GRAD_NORM": 0.5,
        "ANNEAL_LR": False,
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
        help="Dimension of the RAD embeddings"
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

    key = jax.random.PRNGKey(args.seed)

    env = DFABisimEnv()
    env = LogWrapper(env=env, config=config)

    encoder = Encoder(output_dim=args.rad_dim, n_msg_stps=env.sampler.max_size)

    network = ActorCritic(
        action_dim=env.action_space(env.agents[0]).n,
        encoder=encoder
    )

    if config["DEBUG"]:
        key, subkey = jax.random.split(key)
        init_x = env.observation_space(env.agents[0]).sample(subkey)
        key, subkey = jax.random.split(key)
        params = network.init(subkey, init_x)
        summarize_params(params)
    
    train_jit = jax.jit(make_train(config, env, network, _batchify))
    out = train_jit(key)

    os.makedirs(args.save_dir, exist_ok=True)
    trained_params = out["runner_state"][0].params
    trained_encoder_params = {"params": trained_params["params"]["encoder"]}
    with open(f"{args.save_dir}/trained_encoder_ac_params_{args.seed}.msgpack", "wb") as f:
        f.write(serialization.to_bytes(trained_params))
    with open(f"{args.save_dir}/trained_encoder_params_{args.seed}.msgpack", "wb") as f:
        f.write(serialization.to_bytes(trained_encoder_params))

    if config["WANDB"]:
        wandb.finish()

