import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from dfa_samplers import RADSampler, ReachSampler, ReachAvoidSampler
from encoder import Encoder

# Increase font sizes globally.
plt.rcParams.update({'font.size': 24})

dfa2obs = lambda dfa: np.array([int(i) for i in str(dfa.to_int())])

# In-distribution samplers
rad_sampler = RADSampler(p=None, min_size=3, max_size=10)
reach_sampler = ReachSampler(min_size=2, max_size=10)
reach_avoid_sampler = ReachAvoidSampler(p=None, min_size=3, max_size=10)

# OOD samplers
rad_ood_sampler = RADSampler(p=None, min_size=11, max_size=20)
reach_ood_sampler = ReachSampler(min_size=11, max_size=20)
reach_odd_avoid_sampler = ReachAvoidSampler(p=None, min_size=11, max_size=20)

encoder = Encoder(load_file="exps/DFABisimEnv-v1-encoder_1.zip")

# Define number of samples to draw from each sampler.
N = 100  # adjust as needed

dfas = []
# In-distribution samples
for _ in range(N):
    dfas.append(rad_sampler.sample())
for _ in range(N):
    dfas.append(reach_sampler.sample())
for _ in range(N):
    dfas.append(reach_avoid_sampler.sample())
# OOD samples
for _ in range(N):
    dfas.append(rad_ood_sampler.sample())
for _ in range(N):
    dfas.append(reach_ood_sampler.sample())
for _ in range(N):
    dfas.append(reach_odd_avoid_sampler.sample())

# Encode each DFA into its vector representation.
encodings = [encoder.dfa2rad(dfa) for dfa in dfas]

def norm_l2(feat1, feat2):
    feat1 = feat1 / torch.norm(feat1, p=2, dim=-1, keepdim=True)
    feat2 = feat2 / torch.norm(feat2, p=2, dim=-1, keepdim=True)
    d = torch.norm(feat1 - feat2, p=2, dim=-1)
    return d

# Compute the normalized L2 distance matrix between all encoder outputs.
def compute_normalized_distance_matrix(encodings):
    n = len(encodings)
    dists = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            dists[i, j] = norm_l2(encodings[i], encodings[j])
    return dists

distance_matrix = compute_normalized_distance_matrix(encodings)

plt.figure(figsize=(10, 8))
ax = sns.heatmap(distance_matrix, cmap="rocket_r")
plt.title(r"Bisimulation Metrics between DFAs")

# Remove axis labels.
ax.set_xlabel("")
ax.set_ylabel("")

# Calculate tick positions as the midpoints of each group.
# We have 6 groups in order:
#  0 to N-1          : RAD
#  N to 2N-1         : Reach
#  2N to 3N-1        : Reach-Avoid
#  3N to 4N-1        : RAD-OOD
#  4N to 5N-1        : Reach-OOD
#  5N to 6N-1        : Reach-Avoid-OOD
tick_positions = [ (N - 1) / 2,
                   N + (N - 1) / 2,
                   2 * N + (N - 1) / 2,
                   3 * N + (N - 1) / 2,
                   4 * N + (N - 1) / 2,
                   5 * N + (N - 1) / 2 ]
tick_labels = ["RAD", "R", "RA", "RAD\nOOD", "R\nOOD", "RA\nOOD"]

# Set tick positions and labels for both axes.
ax.set_xticks(tick_positions)
ax.set_xticklabels(tick_labels, rotation=0)
ax.set_yticks(tick_positions)
ax.set_yticklabels(tick_labels, rotation=90)

plt.tight_layout()
# Save with reduced margins
plt.savefig("heatmap.png", bbox_inches="tight", pad_inches=0.1)
plt.close()
