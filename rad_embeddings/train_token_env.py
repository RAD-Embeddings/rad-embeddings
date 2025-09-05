import os
import sys
import jax
import yaml
import wandb
import jraph
import distrax
import argparse
import numpy as np
import jax.numpy as jnp
import flax.linen as nn
from ppo import make_train
from encoder import Encoder
from dfax import list2batch, batch2graph
from wrappers import LogWrapper
from flax.core import FrozenDict
from utils import summarize_params
from dfa_gym import TokenEnv, DFAWrapper
import flax.serialization as serialization
from dfax.samplers import ReachSampler, ReachAvoidSampler, ConflictSampler
from flax.linen.initializers import constant, orthogonal


class ActorCritic(nn.Module):
    action_dim: int
    encoder: nn.Module
    encoder_params: FrozenDict
    n_agents: int
    deterministic: bool = False

    def setup(self):
        padding = "VALID"
        self.cnn = nn.Sequential([
            nn.Conv(16, (2, 2), padding=padding, kernel_init=orthogonal(np.sqrt(2))),
            nn.relu,
            nn.Conv(32, (2, 2), padding=padding, kernel_init=orthogonal(np.sqrt(2))),
            nn.relu,
            nn.Conv(64, (2, 2), padding=padding, kernel_init=orthogonal(np.sqrt(2))),
            nn.relu,
            lambda x: x.reshape((x.shape[0], -1)),
        ])
        self.agent_feat = nn.Embed(self.n_agents, 32)
        self.value_net = nn.Sequential([
            nn.Dense(64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)),
            nn.relu,
            nn.Dense(64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)),
            nn.relu,
            nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))
        ])
        self.policy_net = nn.Sequential([
            nn.Dense(64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)),
            nn.relu,
            nn.Dense(64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)),
            nn.relu,
            nn.Dense(64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)),
            nn.relu,
            nn.Dense(self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0))
        ])
        self.task_feat = nn.Sequential([
            nn.Dense(256, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)),
            nn.tanh,
            nn.Dense(256, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)),
            nn.tanh,
            nn.Dense(32, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))
        ])

    @nn.compact
    def __call__(self, batch):
        obs_batch = batch["obs"]
        if obs_batch.ndim == 3: # (C, H, W)
            obs_batch = obs_batch[None, ...] # -> (1, C, H, W)
        elif obs_batch.ndim != 4:
            raise ValueError(f"Expected (C, H, W) or (B, C, H, W), got {obs_batch.shape} for obs")
        obs_batch = jnp.transpose(obs_batch, (0, 2, 3, 1)) # -> (B, H, W, C)
        obs_feat = self.cnn(obs_batch)

        guarantee_batch = batch["guarantee"]
        guarantee_graph = batch2graph(guarantee_batch)
        guarantee_feat = jax.lax.stop_gradient(self.encoder.apply(self.encoder_params, guarantee_graph))

        task_feat = guarantee_feat

        task_feat = jnp.concatenate([obs_feat, task_feat], axis=-1)

        if "assume" in batch:
            batch_size, rad_size = guarantee_feat.shape
            assume_batch = batch["assume"]
            assume_graph = batch2graph(assume_batch)
            assume_feat = jax.lax.stop_gradient(self.encoder.apply(self.encoder_params, assume_graph))
            assume_feat = assume_feat.reshape(batch_size, -1, rad_size).reshape(batch_size, -1)
            task_feat = jnp.concatenate([task_feat, assume_feat], axis=-1)

        if "agent_id" in batch:
            agent_id_batch = batch["agent_id"]
            if agent_id_batch.ndim == 0:
                agent_id_batch = agent_id_batch[None, ...] # -> (1,)
            elif agent_id_batch.ndim != 1:
                raise ValueError(f"Expected () or (B,), got {agent_id_batch.shape} for agent_id")
            agent_feat = self.agent_feat(agent_id_batch)
            task_feat = jnp.concatenate([task_feat, agent_feat], axis=-1)

        task_feat = self.task_feat(task_feat)

        feat = jnp.concatenate([obs_feat, task_feat], axis=-1)

        value = self.value_net(feat)
        logits = self.policy_net(feat)

        if self.deterministic:
            action = jnp.argmax(logits, axis=-1)
            return action, jnp.squeeze(value, axis=-1)
        else:
            pi = distrax.Categorical(logits=logits)
            return pi, jnp.squeeze(value, axis=-1)


def _batchify(obss: dict, agents):

    obs_batch = jnp.stack([obss[agent]["obs"] for agent in agents], axis=0)
    obs_batch = obs_batch if obs_batch.ndim == 4 else jnp.concatenate(obs_batch, axis=0)

    guarantee_batch = list2batch([obss[agent]["guarantee"] for agent in agents])

    obs = {"obs": obs_batch, "guarantee": guarantee_batch}

    if "assume" in obss[agents[0]]:
        assume_batch = list2batch([obss[agent]["assume"] for agent in agents])
        obs["assume"] = assume_batch

    if "agent_id" in obss[agents[0]]:
        agent_id_batch = jnp.stack([obss[agent]["agent_id"] for agent in agents], axis=0)
        agent_id_batch = agent_id_batch if agent_id_batch.ndim == 1 else jnp.concatenate(agent_id_batch, axis=0)
        obs["agent_id"] = agent_id_batch

    return obs


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Train TokenEnv policy")
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed used for PRNGKey (default: 42)"
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Config file"
    )
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    assert config is not None

    config["LOG"] = f"""{config["LOG_FILE_PREFIX"]}_{args.seed}.csv"""

    if config["WANDB"]:
        wandb.init(
            entity=config["WANDB_ENTITY"],
            project=config["WANDB_PROJECT"],
            config=config
        )

    token_env = TokenEnv(
        layout=config["LAYOUT"],
        max_steps_in_episode=config["MAX_EP_LEN"]
    )

    if config["DFA_SAMPLER"] == "Reach":
        sampler = ReachSampler(
            p=config["DFA_SIZE_P"],
            max_size=config["DFA_MAX_SIZE"],
            prob_stutter=config["DFA_PROB_STUTTER"],
            n_tokens=token_env.n_tokens
        )
    elif config["DFA_SAMPLER"] == "ReachAvoid":
        sampler = ReachAvoidSampler(
            p=config["DFA_SIZE_P"],
            max_size=config["DFA_MAX_SIZE"],
            prob_stutter=config["DFA_PROB_STUTTER"],
            n_tokens=token_env.n_tokens
        )
    elif config["DFA_SAMPLER"] == "RAD":
        sampler = RADSampler(
            p=config["DFA_SIZE_P"],
            max_size=config["DFA_MAX_SIZE"],
            prob_stutter=config["DFA_PROB_STUTTER"],
            n_tokens=token_env.n_tokens
        )
    else:
        raise ValueError

    env = DFAWrapper(
        env=token_env,
        sampler=sampler,
        max_eoe_reward=config["MAX_COOP_REWARD"]
    )
    env = LogWrapper(env=env, config=config)

    encoder, encoder_params = Encoder.load_params(
        max_size=env.sampler.max_size,
        encoder_dim=config["ENCODER_DIM"],
        encoder_file=f"""{config["ENCODER_FILE_PREFIX"]}_{args.seed}"""
    )

    network = ActorCritic(
        action_dim=env.action_space(env.agents[0]).n,
        encoder=encoder,
        encoder_params=encoder_params,
        n_agents=env.num_agents
    )

    key = jax.random.PRNGKey(args.seed)

    if config["DEBUG"]:
        key, subkey = jax.random.split(key)
        init_x = env.observation_space(env.agents[0]).sample(subkey)
        key, subkey = jax.random.split(key)
        params = network.init(subkey, init_x)
        summarize_params(params)

    train_jit = jax.jit(make_train(config, env, network, _batchify))
    out = train_jit(key)

    trained_params = out["runner_state"][0].params
    with open(f"""{config["SAVE_FILE_PREFIX"]}_{args.seed}""", "wb") as f:
        f.write(serialization.to_bytes(trained_params))

    if config["WANDB"]:
        wandb.finish()

