import jax
import time
import jraph
import optax
import distrax
import numpy as np
import jax.numpy as jnp
import flax.linen as nn
from flax import struct
from dfa_gym import DFABisimEnv
from collections import deque
from wrappers import LogWrapper
from flax.training.train_state import TrainState
from flax.linen.initializers import constant, orthogonal

from dfax import dfa2dfax, dfax2dfa, batch2graph


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
        graph
    ) -> jnp.ndarray:

        linear_h = nn.Dense(self.hidden_dim, name='linear_h')
        linear_e = nn.Dense(self.hidden_dim, name='linear_e')
        conv = GATv2Conv(out_dim=self.hidden_dim, num_heads=self.n_heads)
        activation = nn.tanh
        g_embed = nn.Dense(self.output_dim, name='g_embed')

        h0 = linear_h(graph["node_features"].astype(jnp.float32))  # [N, hidden_dim]
        e = linear_e(graph["edge_features"].astype(jnp.float32))  # [N, hidden_dim]
        h = h0

        mask = graph["n_states"]

        for _ in range(10): # TODO: use flax.linen.while_loop instead!
            _h = activation(conv(jnp.concatenate([h, h0], axis=-1), e, graph["edge_index"]).sum(axis=1))
            h = jnp.where((mask > 0)[:, None], _h, h)
            mask -= 1

        return g_embed(h[graph["current_state"]])


class ActorCritic(nn.Module):
    action_dim: int

    @nn.compact
    def __call__(self, batch):
        # B = batch["graph_l"]["current_state"].shape[0]  # batch size

        # # Dummy shapes
        # logits = jnp.zeros((B, 10))
        # value = jnp.zeros((B,))

        # pi = distrax.Categorical(logits=logits)
        # return pi, value

        model = Model(input_dim=3, output_dim=32)

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

        feat = model(graph)

        feat_l, feat_r = jnp.array_split(feat, 2)

        feat = jnp.concatenate([feat_l, feat_r], axis=-1)  # shape (B, feat_dim)

        logits = nn.Dense(10, kernel_init=orthogonal(0.01), bias_init=constant(0.0))(feat)
        value = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(feat)

        pi = distrax.Categorical(logits=logits)
        return pi, jnp.squeeze(value, axis=-1)

@struct.dataclass
class Transition():
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray
    info: jnp.ndarray

def batchify(obss: dict, agents):
    return obss[agents[0]]

def unbatchify(actions: jnp.ndarray, agents, n_envs):
    return {agents[0]: actions}

def make_train(config, env):
    config["NUM_AGENTS"] = env.num_agents
    config["NUM_ACTORS"] = config["NUM_AGENTS"] * config["NUM_ENVS"]
    config["NUM_UPDATES"] = (
        config["TOTAL_TIMESTEPS"] // config["NUM_STEPS"] // config["NUM_ENVS"]
    )
    config["MINIBATCH_SIZE"] = (
        config["NUM_ACTORS"] * config["NUM_STEPS"] // config["NUM_MINIBATCHES"]
    )

    def linear_schedule(count):
        frac = (
            1.0
            - (count // (config["NUM_MINIBATCHES"] * config["UPDATE_EPOCHS"]))
            / config["NUM_UPDATES"]
        )
        return config["LR"] * frac

    def train(rng):
        # INIT NETWORK
        network = ActorCritic(env.action_space(env.agents[0]).n)
        rng, _rng = jax.random.split(rng)
        # init_x = jnp.zeros(env.observation_space(env.agents[0]).shape)
        init_x = env.observation_space(env.agents[0]).sample(_rng)
        network_params = network.init(_rng, init_x)
        if config["ANNEAL_LR"]:
            tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(learning_rate=linear_schedule, eps=1e-5),
            )
        else:
            tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(config["LR"], eps=1e-5),
            )
        train_state = TrainState.create(
            apply_fn=network.apply,
            params=network_params,
            tx=tx,
        )

        # INIT ENV
        rng, _rng = jax.random.split(rng)
        reset_rng = jax.random.split(_rng, config["NUM_ENVS"])
        obsv, env_state = jax.vmap(env.reset)(reset_rng)

        # TRAIN LOOP
        def _update_step(runner_state, unused):
            # COLLECT TRAJECTORIES
            def _env_step(runner_state, unused):
                train_state, env_state, last_obs, rng = runner_state

                obs_batch = batchify(last_obs, env.agents)

                # SELECT ACTION
                rng, _rng = jax.random.split(rng)
                pi, value = network.apply(train_state.params, obs_batch)
                action = pi.sample(seed=_rng)
                log_prob = pi.log_prob(action)

                env_act = unbatchify(action, env.agents, config["NUM_ENVS"])

                # STEP ENV
                rng, _rng = jax.random.split(rng)
                rng_step = jax.random.split(_rng, config["NUM_ENVS"])
                # print(action)
                # print(rng_step)
                # print(env_state)
                # print(env_act)
                # input()
                obsv, env_state, reward, done, info = jax.vmap(env.step)(rng_step, env_state, env_act)
                info = jax.tree.map(lambda x: x.reshape((config["NUM_ACTORS"])), info)
                transition = Transition(
                    done=batchify(done, env.agents),
                    action=action,
                    value=value,
                    reward=batchify(reward, env.agents),
                    log_prob=log_prob,
                    obs=obs_batch,
                    info=info
                )
                runner_state = (train_state, env_state, obsv, rng)
                return runner_state, transition

            runner_state, traj_batch = jax.lax.scan(
                _env_step, runner_state, None, config["NUM_STEPS"]
            )

            # CALCULATE ADVANTAGE
            train_state, env_state, last_obs, rng = runner_state
            last_obs_batch = batchify(last_obs, env.agents)
            _, last_val = network.apply(train_state.params, last_obs_batch)

            def _calculate_gae(traj_batch, last_val):
                def _get_advantages(gae_and_next_value, transition):
                    gae, next_value = gae_and_next_value
                    done, value, reward = (
                        transition.done,
                        transition.value,
                        transition.reward,
                    )
                    delta = reward + config["GAMMA"] * next_value * (1 - done) - value
                    gae = (
                        delta
                        + config["GAMMA"] * config["GAE_LAMBDA"] * (1 - done) * gae
                    )
                    return (gae, value), gae

                _, advantages = jax.lax.scan(
                    _get_advantages,
                    (jnp.zeros_like(last_val), last_val),
                    traj_batch,
                    reverse=True,
                    unroll=16,
                )
                return advantages, advantages + traj_batch.value

            advantages, targets = _calculate_gae(traj_batch, last_val)

            # UPDATE NETWORK
            def _update_epoch(update_state, unused):
                def _update_minbatch(train_state, batch_info):
                    traj_batch, advantages, targets = batch_info

                    def _loss_fn(params, traj_batch, gae, targets):
                        # RERUN NETWORK
                        pi, value = network.apply(params, traj_batch.obs)
                        log_prob = pi.log_prob(traj_batch.action)

                        # CALCULATE VALUE LOSS
                        value_pred_clipped = traj_batch.value + (
                            value - traj_batch.value
                        ).clip(-config["CLIP_EPS"], config["CLIP_EPS"])
                        value_losses = jnp.square(value - targets)
                        value_losses_clipped = jnp.square(value_pred_clipped - targets)
                        value_loss = (
                            0.5 * jnp.maximum(value_losses, value_losses_clipped).mean()
                        )

                        # CALCULATE ACTOR LOSS
                        ratio = jnp.exp(log_prob - traj_batch.log_prob)
                        gae = (gae - gae.mean()) / (gae.std() + 1e-8)
                        loss_actor1 = ratio * gae
                        loss_actor2 = (
                            jnp.clip(
                                ratio,
                                1.0 - config["CLIP_EPS"],
                                1.0 + config["CLIP_EPS"],
                            )
                            * gae
                        )
                        loss_actor = -jnp.minimum(loss_actor1, loss_actor2)
                        loss_actor = loss_actor.mean()
                        entropy = pi.entropy().mean()

                        total_loss = (
                            loss_actor
                            + config["VF_COEF"] * value_loss
                            - config["ENT_COEF"] * entropy
                        )
                        return total_loss, (value_loss, loss_actor, entropy)

                    grad_fn = jax.value_and_grad(_loss_fn, has_aux=True)
                    total_loss, grads = grad_fn(
                        train_state.params, traj_batch, advantages, targets
                    )
                    train_state = train_state.apply_gradients(grads=grads)
                    return train_state, total_loss

                train_state, traj_batch, advantages, targets, rng = update_state
                rng, _rng = jax.random.split(rng)
                # Batching and Shuffling
                batch_size = config["MINIBATCH_SIZE"] * config["NUM_MINIBATCHES"]
                assert (
                    batch_size == config["NUM_STEPS"] * config["NUM_ACTORS"]
                ), "batch size must be equal to number of steps * number of envs"
                permutation = jax.random.permutation(_rng, batch_size)
                batch = (traj_batch, advantages, targets)
                batch = jax.tree.map(
                    lambda x: x.reshape((batch_size,) + x.shape[2:]), batch
                )
                shuffled_batch = jax.tree.map(
                    lambda x: jnp.take(x, permutation, axis=0), batch
                )
                # Mini-batch Updates
                minibatches = jax.tree.map(
                    lambda x: jnp.reshape(
                        x, [config["NUM_MINIBATCHES"], -1] + list(x.shape[1:])
                    ),
                    shuffled_batch,
                )
                train_state, total_loss = jax.lax.scan(
                    _update_minbatch, train_state, minibatches
                )
                update_state = (train_state, traj_batch, advantages, targets, rng)
                return update_state, total_loss
            # Updating Training State and Metrics:
            update_state = (train_state, traj_batch, advantages, targets, rng)
            update_state, loss_info = jax.lax.scan(
                _update_epoch, update_state, None, config["UPDATE_EPOCHS"]
            )
            train_state = update_state[0]
            metric = traj_batch.info
            rng = update_state[-1]
            
            # Debugging mode
            if config.get("DEBUG"):
                return_buffer = deque(maxlen=100) # this is fine on the debug side
                disc_return_buffer = deque(maxlen=100) # this is fine on the debug side
                def callback(info):
                    return_values = info["returned_episode_returns"][info["returned_episode"]]
                    return_buffer.extend(return_values)
                    disc_return_values = info["returned_episode_disc_returns"][info["returned_episode"]]
                    disc_return_buffer.extend(disc_return_values)
                    timesteps = info["timestep"][-1, :]
                    global_step = jnp.sum(timesteps) / config["NUM_AGENTS"]
                    mean_return_value = float(np.mean(return_buffer))
                    mean_disc_return_value = float(np.mean(disc_return_buffer))
                    jax.debug.print(f"global step={global_step}, mean return={mean_return_value}, mean disc return={mean_disc_return_value}", ordered=True)
                jax.debug.callback(callback, metric)

            runner_state = (train_state, env_state, last_obs, rng)
            return runner_state, metric

        rng, _rng = jax.random.split(rng)
        runner_state = (train_state, env_state, obsv, _rng)
        runner_state, metric = jax.lax.scan(
            _update_step, runner_state, None, config["NUM_UPDATES"]
        )
        return {"runner_state": runner_state, "metrics": metric}

    return train


if __name__ == "__main__":
    # config = {
    #     "LR": 2.5e-4,
    #     "NUM_ENVS": 4,
    #     "NUM_STEPS": 128,
    #     "TOTAL_TIMESTEPS": 5e3,
    #     "UPDATE_EPOCHS": 4,
    #     "NUM_MINIBATCHES": 4,
    #     "GAMMA": 0.99,
    #     "GAE_LAMBDA": 0.95,
    #     "CLIP_EPS": 0.2,
    #     "ENT_COEF": 0.01,
    #     "VF_COEF": 0.5,
    #     "MAX_GRAD_NORM": 0.5,
    #     "ANNEAL_LR": True,
    #     "DEBUG": False,
    # }
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
    train_jit = jax.jit(make_train(config, env))
    print("Compiled")
    # train_jit = make_train(config, env)
    # start = time.time()
    out = train_jit(rng)
    # end = time.time()
    # print(end - start)
