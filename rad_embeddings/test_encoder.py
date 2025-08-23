import jax
import argparse
from encoder import Encoder
from dfa_gym import DFABisimEnv
import flax.serialization as serialization
from train_encoder import ActorCritic, _batchify


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
    args = parser.parse_args()

    key = jax.random.PRNGKey(args.seed)

    env = DFABisimEnv()
    encoder = Encoder(output_dim=args.rad_dim, n_msg_stps=env.sampler.max_size)
    ac = ActorCritic(action_dim=env.action_space(env.agents[0]).n, encoder=encoder)

    key, subkey = jax.random.split(key)
    init_x = env.observation_space(env.agents[0]).sample(subkey)

    key, subkey = jax.random.split(key)
    ac_params = ac.init(subkey, init_x)

    ac_dir = f"{args.save_dir}/trained_encoder_ac_params_{args.seed}.msgpack"
    with open(ac_dir, "rb") as f:
        ac_params = serialization.from_bytes(ac_params, f.read())

    policy = lambda x: ac.apply(ac_params, _batchify(x, env.agents))

    n = 1_000

    for i in range(n):
        key, subkey = jax.random.split(key)
        obs, state = env.reset(subkey)
        init_state = state
        generated_str = []
        done = False
        while not done:
            pi, value = policy(obs)

            key, subkey = jax.random.split(key)
            action = pi.sample(seed=subkey)
            action = {agent: action[i] for i, agent in enumerate(env.agents)}

            generated_str.append(action["agent_0"])

            key, subkey = jax.random.split(key)
            obs, state, reward, done, info = env.step(subkey, state, action)
            done = done["__all__"]
            if done and reward["agent_0"] < 0:
                print("init_state", init_state)
                print("generated_str", generated_str)
                input()
            
    