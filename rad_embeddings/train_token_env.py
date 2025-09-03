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
    is_circular: bool
    no_assume: bool
    n_agents: int
    deterministic: bool = False

    def setup(self):
        padding = "CIRCULAR" if self.is_circular else "VALID"
        self.cnn = nn.Sequential([
            nn.Conv(16, (2, 2), padding=padding, kernel_init=orthogonal(np.sqrt(2))),
            nn.relu,
            nn.Conv(32, (2, 2), padding=padding, kernel_init=orthogonal(np.sqrt(2))),
            nn.relu,
            nn.Conv(64, (2, 2), padding=padding, kernel_init=orthogonal(np.sqrt(2))),
            nn.relu,
            lambda x: x.reshape((x.shape[0], -1)),
            # nn.Dense(32, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))
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
            nn.relu,
            nn.Dense(128, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)),
            nn.relu,
            nn.Dense(64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))
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

        if not self.no_assume:
            batch_size, rad_size = guarantee_feat.shape
            assume_batch = batch["assume"]
            assume_graph = batch2graph(assume_batch)
            assume_feat = jax.lax.stop_gradient(self.encoder.apply(self.encoder_params, assume_graph))
            assume_feat = assume_feat.reshape(batch_size, -1, rad_size).reshape(batch_size, -1)

            agent_id_batch = batch["agent_id"]
            if agent_id_batch.ndim == 0:
                agent_id_batch = agent_id_batch[None, ...] # -> (1,)
            elif agent_id_batch.ndim != 1:
                raise ValueError(f"Expected () or (B,), got {agent_id_batch.shape} for agent_id")
            agent_feat = self.agent_feat(agent_id_batch)

            env_task_feat = nn.Sequential([
                nn.Dense(256, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)),
                nn.relu,
                nn.Dense(128, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)),
                nn.relu,
                nn.Dense(64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)),
                nn.relu,
                nn.Dense(32, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))
            ])(jnp.concatenate([obs_feat, assume_feat, agent_feat], axis=-1))

            cooperate_logits = nn.Sequential([
                nn.Dense(256, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)),
                nn.relu,
                nn.Dense(128, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)),
                nn.relu,
                nn.Dense(64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)),
                nn.relu,
                nn.Dense(2, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))
            ])(jnp.concatenate([obs_feat, guarantee_feat, assume_feat, guarantee_feat - assume_feat, assume_feat - guarantee_feat, agent_feat], axis=-1))

            cooperate_dist = distrax.Categorical(logits=cooperate_logits)

            cooperate_choice = cooperate_dist.mode() == 0

            task_feat = jnp.where(cooperate_choice[:, None], guarantee_feat, env_task_feat)

        # if not self.no_assume:

        #     task_feat = jnp.concatenate([obs_feat, task_feat], axis=-1)

        #     if "assume" in batch:
        #         batch_size, rad_size = guarantee_feat.shape
        #         assume_batch = batch["assume"]
        #         assume_graph = batch2graph(assume_batch)
        #         assume_feat = jax.lax.stop_gradient(self.encoder.apply(self.encoder_params, assume_graph))
        #         assume_feat = assume_feat.reshape(batch_size, -1, rad_size).reshape(batch_size, -1)
        #         task_feat = jnp.concatenate([task_feat, assume_feat], axis=-1)

        #     if "agent_id" in batch:
        #         agent_id_batch = batch["agent_id"]
        #         if agent_id_batch.ndim == 0:
        #             agent_id_batch = agent_id_batch[None, ...] # -> (1,)
        #         elif agent_id_batch.ndim != 1:
        #             raise ValueError(f"Expected () or (B,), got {agent_id_batch.shape} for agent_id")
        #         agent_feat = self.agent_feat(agent_id_batch)
        #         task_feat = jnp.concatenate([task_feat, agent_feat], axis=-1)

        #     task_feat = self.task_feat(task_feat)

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
    config = {
        "LR": 3e-4,
        "NUM_ENVS": 64,
        "NUM_STEPS": 512,
        "TOTAL_TIMESTEPS": 1e8,
        "UPDATE_EPOCHS": 10,
        "NUM_MINIBATCHES": 8,
        "GAMMA": 0.99,
        "GAE_LAMBDA": 0.95,
        "CLIP_EPS": 0.2,
        "ENT_COEF": 0.01,
        "VF_COEF": 0.5,
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
        help="Size of the RAD embeddings"
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
    parser.add_argument(
        "--n-agents",
        type=int,
        default=1,
        help="Number of agents"
    )
    parser.add_argument(
        "--use-fixed-map",
        action="store_true",
        help="Use fixed map in TokenEnv"
    )
    parser.add_argument(
        "--n-token-repeat",
        type=int,
        default=1,
        help="Number of token repeats in TokenEnv"
    )
    parser.add_argument(
        "--circular",
        action="store_true",
        help="Use circular map in TokenEnv"
    )
    parser.add_argument(
        "--no-assume",
        action="store_true",
        help="Don't pass assume part to the polcy"
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

    # env = DFAWrapper(
    #     TokenEnv(
    #         n_agents=args.n_agents,
    #         fixed_map_seed=args.seed if args.use_fixed_map else None,
    #         n_token_repeat=args.n_token_repeat,
    #         is_circular=args.circular
    #     ),
    #     sampler=ReachAvoidSampler(max_size=6)
    #     # sampler=ConflictSampler(max_size=6, n_agents=args.n_agents)
    # )

    layout = """
        [ # ][ # ][ # ][ # ][ # ][ # ][ # ][ # ][ # ][ # ][ # ][ # ]
        [ # ][   ][   ][   ][   ][   ][   ][ # ][ 0 ][   ][ 1 ][ # ]
        [ # ][   ][   ][ b ][ b ][ b ][   ][ # ][   ][ 4 ][   ][ # ]
        [ # ][   ][   ][ b ][ b ][ b ][   ][ # ][ 3 ][   ][ 2 ][ # ]
        [ # ][   ][   ][ b ][ b ][ b ][   ][ # ][ # ][#,a][ # ][ # ]
        [ # ][ A ][   ][   ][   ][   ][   ][   ][   ][   ][   ][ # ]
        [ # ][ B ][   ][   ][   ][   ][   ][   ][   ][   ][   ][ # ]
        [ # ][   ][   ][ a ][ a ][ a ][   ][ # ][ # ][#,b][ # ][ # ]
        [ # ][   ][   ][ a ][ a ][ a ][   ][ # ][ 5 ][   ][ 6 ][ # ]
        [ # ][   ][   ][ a ][ a ][ a ][   ][ # ][   ][ 9 ][   ][ # ]
        [ # ][   ][   ][   ][   ][   ][   ][ # ][ 8 ][   ][ 7 ][ # ]
        [ # ][ # ][ # ][ # ][ # ][ # ][ # ][ # ][ # ][ # ][ # ][ # ]
    """

    token_env = TokenEnv(layout=layout, max_steps_in_episode=200)

    env = DFAWrapper(
        env=token_env,
        sampler=ReachSampler(max_size=4, prob_stutter=1.0, n_tokens=token_env.n_tokens)
    )
    env = LogWrapper(env=env, config=config)

    encoder, encoder_params = Encoder.load_params(
        output_dim=args.rad_dim,
        n_msg_stps=env.sampler.max_size,
        encoder_dir=f"{args.save_dir}/trained_encoder_params_for_seed_{args.seed}_rad_dim_{args.rad_dim}.msgpack"
    )

    network = ActorCritic(
        action_dim=env.action_space(env.agents[0]).n,
        encoder=encoder,
        encoder_params=encoder_params,
        is_circular=args.circular,
        no_assume=args.no_assume,
        n_agents=env.num_agents
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
    with open(f"{args.save_dir}/trained_token_env_policy_params_for_seed_{args.seed}_rad_dim_{args.rad_dim}_n_agents_{args.n_agents}_use_fixed_map_{args.use_fixed_map}.msgpack", "wb") as f:
        f.write(serialization.to_bytes(trained_params))

    if config["WANDB"]:
        wandb.finish()

