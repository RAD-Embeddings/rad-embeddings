import torch
import random
import numpy as np

import dfa_gym
from dfa import DFA
from dfa_samplers import RADSampler

from encoder import Encoder

def trace(dfa: DFA) -> list[DFA]:
    word = dfa.find_word()
    trace = [dfa]
    for a in word:
        next_dfa = dfa.advance([a]).minimize()
        if next_dfa != dfa:
            dfa = next_dfa
            trace.append(dfa)
    return trace

if __name__ == "__main__":

    SEED = 42

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    env_id = "DFABisimEnv-v1"
    encoder_id = env_id + "-encoder"
    save_dir = "storage"

    # Encoder.train(env_id=env_id, save_dir=save_dir, alg="PPO", reparam=True, id=encoder_id, seed=SEED)

    encoder = Encoder(load_file=f"{save_dir}/{encoder_id}")

    dfa = DFA(
        start=0,
        inputs=range(10),
        label=lambda s: s == 5,
        transition=lambda s, a: s + 1 if s == a and s < 5 else s,
    ).minimize()

    dfas = trace(dfa)
    rads = [encoder.dfa2rad(i) for i in dfas]
    # print(torch.cat([rads[1], rads[0]], dim=1).shape)
    # input()
    dists = [encoder.rad2val(torch.cat([rads[i], rads[i + 1]], dim=1)) for i in range(len(rads) - 1)]
    print(dists)
    input()


    # print(dfa)

    # rad = encoder.dfa2rad(dfa)
    # print(rad)

    # token = encoder.rad2token(rad)
    # print(token)

    sampler = RADSampler(p=None)

    results_1 = []
    results_2 = []
    n = 100
    for i in range(n):
        dfa1 = sampler.sample()
        dfa2 = sampler.sample()

        rad1 = encoder.dfa2rad(dfa1)
        rad2 = encoder.dfa2rad(dfa2)

        val1 = encoder.rad2val(rad1)
        val2 = encoder.rad2val(rad2)
        sum_val = encoder.rad2val(rad1 + rad2)
        dif_val = encoder.rad2val(rad1 - rad2)
        results_1.append(sum_val <= val1 + val2)
        results_2.append(dif_val != 0)
        # print(encoder.rad2val(2*rad1), 2*val1, encoder.rad2val(2*rad1) - 2*val1)
        print(encoder.rad2val(rad1 - rad2), encoder.rad2val(rad2 - rad1))
        input()
        # print(val1, val2, sum_val, sum_val <= val1 + val2, sum_val - (val1 + val2))
    print(sum(results_1)/n)
    print(sum(results_2)/n)




