import jax
import argparse
from encoder import Encoder
from utils import summarize_params
from dfa_gym import TokenEnv, DFAWrapper
import flax.serialization as serialization
from dfax.samplers import ReachAvoidSampler
from train_token_env import ActorCritic, _batchify


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Test DFA encoder")
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
    args = parser.parse_args()

    key = jax.random.PRNGKey(args.seed)

    env = DFAWrapper(
        TokenEnv(
            n_agents=args.n_agents,
            use_fixed_map=args.use_fixed_map
        ),
        sampler=ReachAvoidSampler(max_size=6)
    )

    encoder, encoder_params = Encoder.load_params(
        output_dim=args.rad_dim,
        n_msg_stps=env.sampler.max_size,
        encoder_dir=f"{args.save_dir}/trained_encoder_params_for_seed_{args.seed}_rad_dim_{args.rad_dim}.msgpack"
    )
    ac = ActorCritic(
        action_dim=env.action_space(env.agents[0]).n,
        encoder=encoder,
        encoder_params=encoder_params
    )

    key, subkey = jax.random.split(key)
    init_x = env.observation_space(env.agents[0]).sample(subkey)

    key, subkey = jax.random.split(key)
    ac_params = ac.init(subkey, init_x)

    ac_dir = f"{args.save_dir}/trained_token_env_policy_params_for_seed_{args.seed}_rad_dim_{args.rad_dim}_n_agents_{args.n_agents}_use_fixed_map_{args.use_fixed_map}.msgpack"
    with open(ac_dir, "rb") as f:
        ac_params = serialization.from_bytes(ac_params, f.read())

    summarize_params(ac_params)

    policy = lambda obs, key: ac.apply(ac_params, _batchify(obs, env.agents))[0].sample(seed=key)

    n = 1_000

    for i in range(n):
        key, subkey = jax.random.split(key)
        obs, state = env.reset(subkey)
        init_state = state
        generated_str = []
        done = False
        print("Episode", i)
        step = 0
        while not done:
            keys = jax.random.split(key, env.num_agents + 1)
            key, subkeys =  keys[0], keys[1:]
            actions = policy(obs, subkey)
            actions = {agent: actions[i] for i, agent in enumerate(env.agents)}
            key, subkey = jax.random.split(key)
            obs, state, rewards, dones, infos = env.step(subkey, state, actions)
            done = dones["__all__"]
            print("Step", step)
            env.render(state)
            _rewards = {agent: rewards[agent].item() for agent in rewards}
            _dones = {agent: dones[agent].item() for agent in dones}
            print(_rewards)
            print(_dones)
            if any(reward != 0 for reward in _rewards.values()):
                input()
            step += 1
            
    