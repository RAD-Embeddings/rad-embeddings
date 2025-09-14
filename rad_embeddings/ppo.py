import jax
import time
import wandb
import optax
import distrax
import numpy as np
import pandas as pd
import jax.numpy as jnp
import flax.linen as nn
from flax import struct
from pathlib import Path
from collections import deque, Counter
from flax.training.train_state import TrainState


def batchify(obss: dict, agents):
    return obss[agents[0]]


@struct.dataclass
class Transition():
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray
    info: jnp.ndarray

def make_train(config, env, network):
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
        rng, _rng = jax.random.split(rng)
        init_x = env.observation_space(env.agents[0]).sample(_rng)
        rng, _rng = jax.random.split(rng)
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

                _action = action.reshape((-1, config["NUM_ENVS"]))
                env_act = {agent: _action[i] for i, agent in enumerate(env.agents)}

                # STEP ENV
                rng, _rng = jax.random.split(rng)
                rng_step = jax.random.split(_rng, config["NUM_ENVS"])
                obsv, env_state, reward, done, info = jax.vmap(env.step)(rng_step, env_state, env_act)
                info = jax.tree.map(lambda x: x.reshape((config["NUM_ACTORS"])), info)
                transition = Transition(
                    done=jnp.concatenate([done[agent] for agent in env.agents]),
                    action=action,
                    value=value,
                    reward=jnp.concatenate([reward[agent] for agent in env.agents]),
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

            steps_per_update = config["NUM_ENVS"] * config["NUM_STEPS"]

            if config.get("LOG"):
                ep_len_buffer_log = deque(maxlen=steps_per_update)
                return_buffer_log = deque(maxlen=steps_per_update)
                disc_return_buffer_log = deque(maxlen=steps_per_update)
                start_time_log = time.time()

                def callback(info, loss_info):
                    nonlocal start_time_log

                    elapsed = time.time() - start_time_log

                    log = {}

                    timesteps = info["timestep"][-1, :]
                    timestep = int(np.sum(timesteps) / config["NUM_AGENTS"])
                    log["timestep"] = timestep

                    fps = (steps_per_update / elapsed) if elapsed > 0 else 0.0
                    log["fps"] = np.mean(fps)

                    ep_len_values = info["returned_episode_lengths"][info["returned_episode"]]
                    ep_len_buffer_log.extend(ep_len_values)

                    return_values = info["returned_episode_returns"][info["returned_episode"]]
                    return_buffer_log.extend(return_values)

                    disc_return_values = info["returned_episode_disc_returns"][info["returned_episode"]]
                    disc_return_buffer_log.extend(disc_return_values)

                    log["ep_len_min"] = np.min(ep_len_buffer_log)
                    log["ep_len_mean"] = np.mean(ep_len_buffer_log)
                    log["ep_len_max"] = np.max(ep_len_buffer_log)
                    log["ep_len_std"] = np.std(ep_len_buffer_log)

                    log["return_min"] = np.min(return_buffer_log)
                    log["return_mean"] = np.mean(return_buffer_log)
                    log["return_max"] = np.max(return_buffer_log)
                    log["return_std"] = np.std(return_buffer_log)

                    log["disc_return_mean"] = np.mean(disc_return_buffer_log)

                    total_loss, (value_loss, actor_loss, entropy) = loss_info

                    log["total_loss"] = np.mean(total_loss)
                    log["value_loss"] = np.mean(value_loss)
                    log["actor_loss"] = np.mean(actor_loss)
                    log["entropy"] = np.mean(entropy)

                    returns = [float(x.item()) for x in return_buffer_log]
                    n = len(returns)
                    counts = Counter(returns)
                    return_dist = {i: float(counts[i])/float(n) for i in counts}
                    log["min_return_rate"] = return_dist[np.min(returns)]
                    log["max_return_rate"] = return_dist[np.max(returns)]

                    log_file = Path(config.get("LOG"))
                    df = pd.DataFrame([log])
                    df.to_csv(
                        log_file,
                        mode="a",
                        header=not log_file.exists(),
                        index=False
                    )

                    start_time_log = time.time()
                jax.experimental.io_callback(callback, None, metric, loss_info)

            if config.get("WANDB"):
                ep_len_buffer_wandb = deque(maxlen=steps_per_update)
                return_buffer_wandb = deque(maxlen=steps_per_update)
                disc_return_buffer_wandb = deque(maxlen=steps_per_update)
                start_time_wandb = time.time()

                def callback(info, loss_info):
                    nonlocal start_time_wandb

                    elapsed = time.time() - start_time_wandb
                    fps = (steps_per_update / elapsed) if elapsed > 0 else 0.0

                    log = {
                        "fps": np.mean(fps),
                    }

                    ep_len_values = info["returned_episode_lengths"][info["returned_episode"]]
                    ep_len_buffer_wandb.extend(ep_len_values)

                    return_values = info["returned_episode_returns"][info["returned_episode"]]
                    return_buffer_wandb.extend(return_values)

                    disc_return_values = info["returned_episode_disc_returns"][info["returned_episode"]]
                    disc_return_buffer_wandb.extend(disc_return_values)

                    log["ep_len_min"] = np.min(ep_len_buffer_wandb)
                    log["ep_len_mean"] = np.mean(ep_len_buffer_wandb)
                    log["ep_len_max"] = np.max(ep_len_buffer_wandb)
                    log["ep_len_std"] = np.std(ep_len_buffer_wandb)

                    log["return_min"] = np.min(return_buffer_wandb)
                    log["return_mean"] = np.mean(return_buffer_wandb)
                    log["return_max"] = np.max(return_buffer_wandb)
                    log["return_std"] = np.std(return_buffer_wandb)

                    log["disc_return_mean"] = np.mean(disc_return_buffer_wandb)

                    total_loss, (value_loss, actor_loss, entropy) = loss_info

                    log["total_loss"] = np.mean(total_loss)
                    log["value_loss"] = np.mean(value_loss)
                    log["actor_loss"] = np.mean(actor_loss)
                    log["entropy"] = np.mean(entropy)

                    returns = [float(x.item()) for x in return_buffer_wandb]
                    n = len(returns)
                    counts = Counter(returns)
                    return_dist = {i: float(counts[i])/float(n) for i in counts}
                    log["min_return_rate"] = return_dist[np.min(returns)]
                    log["max_return_rate"] = return_dist[np.max(returns)]

                    timesteps = info["timestep"][-1, :]
                    timestep = int(np.sum(timesteps) / config["NUM_AGENTS"])

                    wandb.log(log, step=timestep)

                    start_time_wandb = time.time()
                jax.experimental.io_callback(callback, None, metric, loss_info)
            
            # Debugging mode
            if config.get("DEBUG"):
                ep_len_buffer_debug = deque(maxlen=steps_per_update)
                return_buffer_debug = deque(maxlen=steps_per_update)
                disc_return_buffer_debug = deque(maxlen=steps_per_update)
                start_time_debug = time.time()

                def callback(info, loss_info):
                    nonlocal start_time_debug

                    elapsed = time.time() - start_time_debug
                    fps = (steps_per_update / elapsed) if elapsed > 0 else 0.0

                    log = {
                        "fps": np.mean(fps),
                    }

                    ep_len_values = info["returned_episode_lengths"][info["returned_episode"]]
                    ep_len_buffer_debug.extend(ep_len_values)

                    return_values = info["returned_episode_returns"][info["returned_episode"]]
                    return_buffer_debug.extend(return_values)

                    disc_return_values = info["returned_episode_disc_returns"][info["returned_episode"]]
                    disc_return_buffer_debug.extend(disc_return_values)

                    log["ep_len_min"] = np.min(ep_len_buffer_debug)
                    log["ep_len_mean"] = np.mean(ep_len_buffer_debug)
                    log["ep_len_max"] = np.max(ep_len_buffer_debug)
                    log["ep_len_std"] = np.std(ep_len_buffer_debug)

                    log["return_min"] = np.min(return_buffer_debug)
                    log["return_mean"] = np.mean(return_buffer_debug)
                    log["return_max"] = np.max(return_buffer_debug)
                    log["return_std"] = np.std(return_buffer_debug)

                    log["disc_return_mean"] = np.mean(disc_return_buffer_debug)

                    total_loss, (value_loss, actor_loss, entropy) = loss_info

                    log["total_loss"] = np.mean(total_loss)
                    log["value_loss"] = np.mean(value_loss)
                    log["actor_loss"] = np.mean(actor_loss)
                    log["entropy"] = np.mean(entropy)

                    timesteps = info["timestep"][-1, :]
                    log["timestep"] = int(np.sum(timesteps) / config["NUM_AGENTS"])

                    returns = [float(x.item()) for x in return_buffer_debug]
                    n = len(returns)
                    counts = Counter(returns)
                    return_dist = {i: float(counts[i])/float(n) for i in counts}
                    log["min_return_rate"] = return_dist[np.min(returns)]
                    log["max_return_rate"] = return_dist[np.max(returns)]

                    jax.debug.print(
                        """
timestep         = {timestep}
disc_return_mean = {disc_return_mean}
return_min       = {return_min}
return_mean      = {return_mean}
return_max       = {return_max}
return_std       = {return_std}
ep_len_min       = {ep_len_min}
ep_len_mean      = {ep_len_mean}
ep_len_max       = {ep_len_max}
ep_len_std       = {ep_len_std}
total_loss       = {total_loss}
value_loss       = {value_loss}
actor_loss       = {actor_loss}
entropy          = {entropy}
fps              = {fps}
min_return_rate  = {min_return_rate}
max_return_rate  = {max_return_rate}
return_dist      = {return_dist}
                        """,
                        timestep=log["timestep"],
                        disc_return_mean=log["disc_return_mean"],
                        return_min=log["return_min"],
                        return_mean=log["return_mean"],
                        return_max=log["return_max"],
                        return_std=log["return_std"],
                        ep_len_min=log["ep_len_min"],
                        ep_len_mean=log["ep_len_mean"],
                        ep_len_max=log["ep_len_max"],
                        ep_len_std=log["ep_len_std"],
                        total_loss=log["total_loss"],
                        value_loss=log["value_loss"],
                        actor_loss=log["actor_loss"],
                        entropy=log["entropy"],
                        fps=log["fps"],
                        min_return_rate=log["min_return_rate"],
                        max_return_rate=log["max_return_rate"],
                        return_dist=return_dist,
                        ordered=True)

                    start_time_debug = time.time()

                jax.debug.callback(callback, metric, loss_info)

            runner_state = (train_state, env_state, last_obs, rng)
            return runner_state, metric

        rng, _rng = jax.random.split(rng)
        runner_state = (train_state, env_state, obsv, _rng)
        runner_state, metric = jax.lax.scan(
            _update_step, runner_state, None, config["NUM_UPDATES"]
        )
        return {"runner_state": runner_state, "metrics": metric}

    return train

