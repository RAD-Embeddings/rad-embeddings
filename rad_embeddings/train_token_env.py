import jax
import jraph
import distrax
import numpy as np
import jax.numpy as jnp
import flax.linen as nn
from ppo import make_train
from dfax import batch2graph
from dfa_gym import TokenEnv
from wrappers import LogWrapper
from flax.linen.initializers import constant, orthogonal


class TokenEnvFeaturesExtractor(nn.Module):

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

    @nn.compact
    def __call__(self, obs):
        if obs.ndim == 3: # (C, H, W)
            obs = obs[None, ...] # -> (1, C, H, W)
        elif obs.ndim != 4:
            raise ValueError(f"Expected (C, H, W) or (B, C, H, W), got {obs.shape}")
        obs = jnp.transpose(obs, (0, 2, 3, 1)) # -> (B, H, W, C)
        feat = TokenEnvFeaturesExtractor()(obs)

        policy_hidden = MLP([64, 64, 64])(feat)
        value_hidden = MLP([64, 64])(feat)

        logits = nn.Dense(self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0))(policy_hidden)
        value = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(value_hidden)

        pi = distrax.Categorical(logits=logits)
        return pi, jnp.squeeze(value, axis=-1)


def _batchify(obss: dict, agents):
    shape_without_batch = obss[agents[0]].shape[1:]
    obss_batch = jnp.stack([obss[agent] for agent in agents])
    return obss_batch.reshape((-1, *shape_without_batch))

def _unbatchify(actions: jnp.ndarray, agents, n_envs):
    _actions = actions.reshape((len(agents), n_envs, -1)).squeeze()
    return {agent: _actions[i] for i, agent in enumerate(agents)}


if __name__ == "__main__":
    config = {
        "LR": 2.5e-4,
        "NUM_ENVS": 4,
        "NUM_STEPS": 128,
        "TOTAL_TIMESTEPS": 5e5,
        "UPDATE_EPOCHS": 4,
        "NUM_MINIBATCHES": 4,
        "GAMMA": 0.99,
        "GAE_LAMBDA": 0.95,
        "CLIP_EPS": 0.2,
        "ENT_COEF": 0.01,
        "VF_COEF": 0.5,
        "MAX_GRAD_NORM": 0.5,
        "ANNEAL_LR": True,
        "DEBUG": True,
    }
    env = TokenEnv()
    env = LogWrapper(env=env, config=config)
    rng = jax.random.PRNGKey(30)
    network = ActorCritic(env.action_space(env.agents[0]).n)
    train_jit = jax.jit(make_train(config, env, network, _batchify, _unbatchify))
    out = train_jit(rng)
