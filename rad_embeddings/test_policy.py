import jax
import yaml
import argparse
from encoder import Encoder
from utils import summarize_params
from dfa_gym import TokenEnv, DFAWrapper
import flax.serialization as serialization
from dfax.samplers import ReachSampler, ReachAvoidSampler, ConflictSampler
from train_policy import ActorCritic, _batchify
from collections import Counter


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

    key = jax.random.PRNGKey(16)

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
        max_coop_reward=config["MAX_COOP_REWARD"]
    )

    encoder, encoder_params = Encoder.load_params(
        max_size=env.sampler.max_size,
        encoder_dim=config["ENCODER_DIM"],
        encoder_file=f"""{config["ENCODER_FILE_PREFIX"]}_{args.seed}"""
    )

    ac = ActorCritic(
        action_dim=env.action_space(env.agents[0]).n,
        encoder=encoder,
        encoder_params=encoder_params,
        n_agents=env.num_agents
    )

    key, subkey = jax.random.split(key)
    init_x = env.observation_space(env.agents[0]).sample(subkey)

    key, subkey = jax.random.split(key)
    ac_params = ac.init(subkey, init_x)

    ac_dir = f"""{config["SAVE_FILE_PREFIX"]}_{args.seed}"""
    with open(ac_dir, "rb") as f:
        ac_params = serialization.from_bytes(ac_params, f.read())

    summarize_params(ac_params)

    policy = lambda obs, key: ac.apply(ac_params, _batchify(obs, env.agents))[0].sample(seed=key)
    # policy = lambda obs, key: ac.apply(ac_params, _batchify(obs, env.agents))[0]

    n = 100
    agent_rewards = {agent: [] for agent in env.agents}

    for i in range(n):
        key, subkey = jax.random.split(key)
        obs, state = env.reset(subkey)
        init_state = state
        # env.render(state)
        generated_str = []
        done = False
        # print("Episode", i)
        step = 0
        for agent in env.agents:
            agent_rewards[agent].append(0)
        while not done:
            keys = jax.random.split(key, env.num_agents + 1)
            key, subkeys =  keys[0], keys[1:]
            actions = policy(obs, subkey)
            actions = {agent: actions[i] for i, agent in enumerate(env.agents)}
            key, subkey = jax.random.split(key)
            obs, state, rewards, dones, infos = env.step(subkey, state, actions)
            done = dones["__all__"]
            # print("Step", step)
            # print("actions", actions)
            # env.render(state)
            _rewards = {agent: rewards[agent].item() for agent in rewards}
            _dones = {agent: dones[agent].item() for agent in dones}
            # print(_rewards)
            # print(_dones)
            # jax.numpy.set_printoptions(threshold=10000)
            # print(obs)
            # if any(rewards[agent] <= 0 and dones[agent] for agent in env.agents):
            # input()
            step += 1
            for agent in env.agents:
                agent_rewards[agent][-1] += rewards[agent].item()

        print(f"Test completed for {i + 1} samples.")

        agent_reward_counts = {agent: Counter(agent_rewards[agent]) for agent in env.agents}
        agent_reward_dist = {agent: {count: agent_reward_counts[agent][count]/(i + 1) for count in agent_reward_counts[agent]} for agent in env.agents}
        for agent in env.agents:
            print(agent, agent_reward_dist[agent])

    print(f"Test completed for {args.n} samples.")

    agent_reward_counts = {agent: Counter(agent_rewards[agent]) for agent in env.agents}
    agent_reward_dist = {agent: {i: float(agent_reward_counts[agent][i])/float(n) for i in agent_reward_counts[agent]} for agent in env.agents}
    for agent in env.agents:
        print(agent, agent_reward_dist[agent])

    