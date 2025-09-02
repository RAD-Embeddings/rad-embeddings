import jax
import argparse
from encoder import Encoder
from utils import summarize_params
from dfa_gym import TokenEnv, DFAWrapper
import flax.serialization as serialization
from dfax.samplers import ReachSampler, ReachAvoidSampler, ConflictSampler
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

    key = jax.random.PRNGKey(args.seed)

    # env = DFAWrapper(
    #     TokenEnv(
    #         n_agents=args.n_agents,
    #         fixed_map_seed=args.seed if args.use_fixed_map else None
    #     ),
    #     sampler=ReachAvoidSampler(max_size=6)
    # )

    layout = """
        [ # ][ # ][ # ][ # ][ # ][ # ][ # ][ # ][ # ][ # ][ # ][ # ]
        [ # ][   ][   ][   ][   ][   ][   ][ # ][ 0 ][   ][ 1 ][ # ]
        [ # ][   ][   ][   ][   ][   ][   ][ # ][   ][ 4 ][   ][ # ]
        [ # ][   ][ b ][   ][   ][   ][   ][ # ][ 3 ][   ][ 2 ][ # ]
        [ # ][   ][   ][   ][   ][   ][   ][ # ][ # ][#,a][ # ][ # ]
        [ # ][ A ][   ][   ][   ][   ][   ][   ][   ][   ][   ][ # ]
        [ # ][ B ][   ][   ][   ][   ][   ][   ][   ][   ][   ][ # ]
        [ # ][   ][   ][   ][   ][   ][   ][ # ][ # ][#,b][ # ][ # ]
        [ # ][   ][ a ][   ][   ][   ][   ][ # ][ 5 ][   ][ 6 ][ # ]
        [ # ][   ][   ][   ][   ][   ][   ][ # ][   ][ 9 ][   ][ # ]
        [ # ][   ][   ][   ][   ][   ][   ][ # ][ 8 ][   ][ 7 ][ # ]
        [ # ][ # ][ # ][ # ][ # ][ # ][ # ][ # ][ # ][ # ][ # ][ # ]
    """

    token_env = TokenEnv(layout=layout)

    env = DFAWrapper(
        env=token_env,
        sampler=ReachSampler(max_size=4, prob_stutter=1.0, n_tokens=token_env.n_tokens)
    )

    encoder, encoder_params = Encoder.load_params(
        output_dim=args.rad_dim,
        n_msg_stps=env.sampler.max_size,
        encoder_dir=f"{args.save_dir}/trained_encoder_params_for_seed_{args.seed}_rad_dim_{args.rad_dim}.msgpack"
    )
    ac = ActorCritic(
        action_dim=env.action_space(env.agents[0]).n,
        encoder=encoder,
        encoder_params=encoder_params,
        is_circular=args.circular,
        no_assume=args.no_assume,
        n_agents=env.num_agents
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
        env.render(state)
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
            print("actions", actions)
            env.render(state)
            _rewards = {agent: rewards[agent].item() for agent in rewards}
            _dones = {agent: dones[agent].item() for agent in dones}
            print(_rewards)
            print(_dones)
            # if any(rewards[agent] <= 0 and dones[agent] for agent in env.agents):
            input()
            step += 1
            
    