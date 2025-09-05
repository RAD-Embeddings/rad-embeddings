import jax
import argparse
from encoder import Encoder
from dfa_gym import DFABisimEnv
from dfax.samplers import RADSampler
import flax.serialization as serialization
from train_encoder import ActorCritic, _batchify


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Test DFA encoder")
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
        default="encoder_storage",
        help="Directory for saving the trained encoder (default: encoder_storage)"
    )
    parser.add_argument(
        "--n",
        type=int,
        default=100,
        help="Number of samples (default: 100)"
    )
    parser.add_argument(
        "--rad-dim",
        type=int,
        default=32,
        help="Dimension of the RAD embeddings (default: 32)"
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
        "--deterministic",
        action="store_true",
        help="Run policy deterministically"
    )
    args = parser.parse_args()

    key = jax.random.PRNGKey(args.seed)

    sampler = RADSampler(max_size=args.max_size, n_tokens=args.n_tokens)
    env = DFABisimEnv(sampler=sampler)

    encoder = Encoder(encoder_dim=args.rad_dim, max_size=args.max_size)
    ac = ActorCritic(action_dim=env.action_space(env.agents[0]).n, encoder=encoder, deterministic=args.deterministic)

    key, subkey = jax.random.split(key)
    init_x = env.observation_space(env.agents[0]).sample(subkey)

    key, subkey = jax.random.split(key)
    ac_params = ac.init(subkey, init_x)

    ac_dir = f"{args.save_dir}/encoder_ac_rad_dim_{args.rad_dim}_max_size_{args.max_size}_n_tokens_{args.n_tokens}_params_{args.seed}"
    with open(ac_dir, "rb") as f:
        ac_params = serialization.from_bytes(ac_params, f.read())

    total_reward = 0
    accept_count = 0
    reject_count = 0
    undecide_count = 0
    for i in range(args.n):
        key, subkey = jax.random.split(key)
        obs, state = env.reset(subkey)
        init_state = state
        generated_str = []
        done = False
        while not done:
            if args.deterministic:
                action, _ = ac.apply(ac_params, _batchify(obs, env.agents))
            else:
                key, subkey = jax.random.split(key)
                pi, _ = ac.apply(ac_params, _batchify(obs, env.agents))
                action = pi.sample(seed=subkey)
            action = {agent: action[i] for i, agent in enumerate(env.agents)}
            generated_str.append(action["agent_0"])

            key, subkey = jax.random.split(key)
            obs, state, reward, done, info = env.step(subkey, state, action)
            done = done["__all__"]
            total_reward += reward["agent_0"]
            if done:
                if reward["agent_0"] == 1:
                    accept_count += 1
                elif reward["agent_0"] == -1:
                    reject_count += 1
                else:
                    undecide_count += 1

        print(f"Test completed for {i + 1} samples.", end="\r")

    print(f"Test completed for {args.n} samples.")
    print(f"Mean reward:", total_reward/args.n)
    print(f"Accept rate:", accept_count/args.n)
    print(f"Reject rate:", reject_count/args.n)
    print(f"Undecided rate:", undecide_count/args.n)

