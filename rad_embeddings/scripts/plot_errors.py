import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from dfa_samplers import RADSampler, ReachSampler, ReachAvoidSampler
from encoder import Encoder

# Increase font sizes globally.
plt.rcParams.update({'font.size': 16})

encoder = Encoder(load_file="exps/DFABisimEnv-v1-encoder_1.zip")
# encoder = Encoder(load_file="exps_baseline/DFAEnv-v1-encoder_16.zip")

# Define the list of samplers with names.
# In this example, we use three variations of RADSampler.
sampler_list = [
    # ("RAD", RADSampler(min_size=3, max_size=10, p=None)),
    ("RAD-OOD", RADSampler(min_size=11, max_size=20, p=None)),
    # ("RAD-OOD-Extra", RADSampler(min_size=21, max_size=30, p=None)),
    # ("R", ReachSampler(min_size=3, max_size=10)),
    # ("R-OOD", ReachSampler(min_size=11, max_size=20)),
    # ("RA", ReachAvoidSampler(min_size=3, max_size=10, p=None)),
    # ("RA-OOD", ReachAvoidSampler(min_size=11, max_size=20, p=None)),
]

N = 100  # Number of samples per sampler

# Gather all samples (DFAs and their encodings) across all sampler classes.
all_dfas = []
all_encodings = []
class_indices = {}  # Dictionary mapping sampler name to list of indices in the combined list.

current_index = 0
for name, sampler in sampler_list:
    indices = []
    for _ in range(N):
        dfa = sampler.sample()
        all_dfas.append(dfa)
        all_encodings.append(encoder.dfa2rad(dfa))
        indices.append(current_index)
        current_index += 1
    class_indices[name] = indices

def compute_error_rate_for_class(class_idx, all_dfas, all_encodings, tol=1e-8):
    """
    For a given list of indices corresponding to a sampler class,
    compute the error rate by comparing each sample's encoding against all other samples.
    A collision is counted if two distinct DFAs have nearly identical encodings.
    """
    collision_count = 0
    comparisons = 0
    total = len(all_dfas)
    for i in class_idx:
        for j in range(total):
            if i == j:
                continue
            comparisons += 1
            if all_dfas[i] != all_dfas[j] and np.allclose(all_encodings[i], all_encodings[j], atol=tol):
                collision_count += 1
    return collision_count / comparisons if comparisons > 0 else 0

# Compute the success rate (1 - error rate) per sampler class.
success_rates = {}
for name, indices in class_indices.items():
    error_rate = compute_error_rate_for_class(indices, all_dfas, all_encodings, tol=1e-8)
    success_rate = 1 - error_rate
    success_rates[name] = success_rate
    print(f"Success rate for {name}: {success_rate}")

# Plot a bar graph of the success rates.
labels = list(success_rates.keys())
rates = list(success_rates.values())

plt.figure(figsize=(8, 6))
bars = plt.bar(labels, rates, color="skyblue")
plt.title("Success Rates for Each Sampler (Compared Against All Samples)")
plt.xlabel("Sampler")
plt.ylabel("Success Rate")
plt.ylim(0, 1.1)
plt.tight_layout()
plt.savefig("success_rates.pdf")
plt.close()
