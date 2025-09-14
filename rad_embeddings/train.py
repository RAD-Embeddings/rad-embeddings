import os
import jax
import wandb
import argparse
from ppo import make_train
from dfa_gym import DFABisimEnv
from wrappers import LogWrapper
from dfax.samplers import RADSampler
import flax.serialization as serialization
from flax.traverse_util import flatten_dict
from encoder import EncoderModule, ActorCritic


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
        help="Seed used for PRNGKey (default: 42)"
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default="storage",
        help="Directory for saving the trained encoder (default: storage)"
    )
    parser.add_argument(
        "--max-size",
        type=int,
        default=10,
        help="Number of DFA states (default: 10)"
    )
    parser.add_argument(
        "--n-tokens",
        type=int,
        default=10,
        help="Number tokens (default: 10)"
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
        "--log",
        type=str,
        default="log.csv",
        help="Log csv name (default: log.csv)"
    )
    args = parser.parse_args()

    config["DEBUG"] = args.debug
    config["WANDB"] = args.wandb
    config["LOG"] = args.log

    if config["WANDB"]:
        wandb.init(
            entity="beyazit-y-berkeley-eecs",
            project="rad-jax",
            config=config
        )

    key = jax.random.PRNGKey(args.seed)

    sampler = RADSampler(max_size=args.max_size, n_tokens=args.n_tokens)
    env = DFABisimEnv(sampler=sampler)
    env = LogWrapper(env=env, config=config)

    encoder = EncoderModule(max_size=args.max_size)

    network = ActorCritic(
        action_dim=env.action_space(env.agents[0]).n,
        encoder=encoder
    )

    if config["DEBUG"]:
        key, subkey = jax.random.split(key)
        init_x = env.observation_space(env.agents[0]).sample(subkey)
        key, subkey = jax.random.split(key)
        params = network.init(subkey, init_x)
        flat = flatten_dict(params, sep="/")
        total = 0
        for k, v in flat.items():
            count = v.size
            total += count
            print(f"{k:60} {v.shape} {v.dtype} ({count:,} params)")
        print(f"\nTotal parameters: {total:,}")
    
    train_jit = jax.jit(make_train(config, env, network))
    out = train_jit(key)

    os.makedirs(args.save_dir, exist_ok=True)

    trained_params = out["runner_state"][0].params
    with open(f"{args.save_dir}/encoder_params_max_size_{args.max_size}_n_tokens_{args.n_tokens}_seed_{args.seed}.msgpack", "wb") as f:
        f.write(serialization.to_bytes(trained_params))

    if config["WANDB"]:
        wandb.finish()

