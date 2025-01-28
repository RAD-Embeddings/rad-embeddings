import torch
from dfa import DFA
from dfa_samplers import DFASampler
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from dfa_samplers import ReachSampler, ReachAvoidSampler, RADSampler

from encoder import Encoder

def advance(dfa: DFA) -> DFA:
    word = dfa.find_word()
    sub_word = word[:np.random.randint((len(word) + 1) // 2)]
    return dfa.advance(sub_word).minimize()

def accept(dfa: DFA) -> DFA:
    dfa = DFA(start=0, inputs=dfa.inputs, label=lambda s: True, transition=lambda s, c: s)
    return dfa.minimize()

def reject(dfa: DFA) -> DFA:
    dfa = DFA(start=0, inputs=dfa.inputs, label=lambda s: False, transition=lambda s, c: s)
    return dfa.minimize()

def trace(dfa: DFA) -> list[DFA]:
    word = dfa.find_word()
    trace = [dfa]
    for a in word:
        next_dfa = dfa.advance([a]).minimize()
        if next_dfa != dfa:
            dfa = next_dfa
            trace.append(dfa)
    return trace

n = 2000
k = 5

n_tokens = 10
max_size = 10
dfa_encoder = Encoder("storage/DFABisimEnv-v1-encoder.zip")
# dfa_encoder = Model(15, 32)
# dfa_encoder.load_state_dict(torch.load("temp3_state_dict.pth", weights_only=True))
# for param in dfa_encoder.parameters():
#     param.requires_grad = False
# dfa_encoder.eval()

dfa2obs = lambda dfa: np.array([int(i) for i in str(dfa.to_int())])

rad_sampler = RADSampler(n_tokens=n_tokens, max_size=max_size, p=0.5)
reach_sampler = ReachSampler(n_tokens=n_tokens, max_size=max_size)
reach_avoid_sampler = ReachAvoidSampler(n_tokens=n_tokens, max_size=max_size)

rad_gen = lambda: rad_sampler.sample()
rad_adv_gen = lambda: advance(rad_sampler.sample())

reach_gen = lambda: reach_sampler.sample()
reach_adv_gen = lambda: advance(reach_sampler.sample())

reach_avoid_gen = lambda: reach_avoid_sampler.sample()
reach_avoid_adv_gen = lambda: advance(reach_avoid_sampler.sample())

accept_gen = lambda: accept(rad_sampler.sample())
reject_gen = lambda: accept(rad_sampler.sample())

dfas = [(rad_gen(), "rad", "init") for _ in range(n * k)]
dfas += [(rad_adv_gen(), "rad", "adv") for _ in range(n)]
dfas += [(reach_gen(), "reach", "init") for _ in range(n)]
dfas += [(reach_adv_gen(), "reach", "adv") for _ in range(n)]
dfas += [(reach_avoid_gen(), "reach_avoid", "init") for _ in range(n)]
dfas += [(reach_avoid_adv_gen(), "reach_avoid", "adv") for _ in range(n)]
dfas += [(accept_gen(), "accept", "accept") for _ in range(n // k)]
dfas += [(reject_gen(), "reject", "reject") for _ in range(n // k)]


# reach_sampler = ReachSampler(n_tokens=n_tokens, max_size=6, p=None)

# reach_gen = lambda: dfa2obs(reach_sampler.sample())
# reach_adv_gen = lambda: dfa2obs(advance(reach_sampler.sample()))
# accept_gen = lambda: dfa2obs(accept(reach_sampler.sample()))

# dfas = [(reach_gen(), "reach", "init") for _ in range(n)]
# dfas += [(reach_adv_gen(), "reach", "adv") for _ in range(n)]

# dfa = DFA(
#     start=0,
#     inputs=range(n_tokens),
#     label=lambda s: s == 5,
#     transition=lambda s, a: s + 1 if s == a and s < 5 else s,
# ).minimize()

# dfas += [(accept_gen(), "accept", "accept") for _ in range(n // k)]
# dfas += [(d, "trace", "s" + str(i)) for i, d in enumerate(trace(dfa))]

# np.random.shuffle(dfas)

dfas, hue, style = zip(*dfas)

rads = np.array([dfa_encoder.dfa2rad(dfa) for dfa in dfas]).squeeze()

rads_2d = TSNE(n_components=2, learning_rate='auto', init='random', perplexity=3).fit_transform(rads)

palette = sns.color_palette("Set2")
plt.figure(figsize=(8, 8))
sns.scatterplot(x=rads_2d[:, 0], y=rads_2d[:, 1], hue=hue, style=style, palette=palette, alpha=0.5)
plt.xlabel("1st T-SNE Dimension")
plt.ylabel("2nd T-SNE Dimension")
plt.xticks([])
plt.yticks([])
plt.legend(ncol=3, bbox_to_anchor=(0.5, -0.16), loc='lower center')
plt.tight_layout()
plt.savefig("temp.pdf", bbox_inches='tight')
