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
dfa_encoder = Encoder("storage/DFABisimEnv-v1-metric_norml2_gamma_90_ent_0-encoder.zip")
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
























# def get_projection(x, method):
#     if method == "tsne":
#         return TSNE(n_components=2, learning_rate='auto', init='random', perplexity=3).fit_transform(x)
#     elif method == "pca":
#         return PCA(n_components=2).fit_transform(x)
#     elif method == "spectral":
#         return spectral_embedding(distance_matrix(x, x), n_components=2)
#     # model = make_pipeline(StandardScaler(), TSNE(n_components=3, learning_rate='auto', init='random', perplexity=3))
#     # model = TSNE(n_components=3, learning_rate='auto', init='random', perplexity=3)
#     # model = umap.UMAP(n_components=3)
#     # return spectral_embedding(distance_matrix(x, x), n_components=2)
#     return None

# sampler_names = ["Reach-Avoid Derived", "Reach-Avoid", "Reach", "Reach-Avoid with Redemption", "Parity"]

# x = []
# x_hue = []
# x_style = []
# # x_size = []
# for sampler_name in sampler_names:
#     if sampler_name == "Reach-Avoid Derived" and plot_type == "scatter":
#         old_n = n
#         n *= 4
#     sampler_id = sampler_name.replace(" ", "").replace("-", "").replace("with", "")
#     sampler = getDFASampler("Compositional" + sampler_id + "_2_2_4_4", propositions)
#     samples = [sampler.sample() for _ in range(n)]
#     two_collapsed_samples = [collapse_conjunctions(dfa_goal, k=2) for dfa_goal in samples]
#     one_step_advance_samples = [advance_dfas(dfa_goal, k=1) for dfa_goal in samples]

#     x.extend([dfa_builder(d) for d in samples])
#     x_hue.extend([sampler_name]*n)
#     x_style.extend(["No Op"]*n)

#     if plot_type == "scatter":

#         x.extend([dfa_builder(d) for d in two_collapsed_samples])
#         x_hue.extend([sampler_name]*n)
#         x_style.extend(["2-Conjunction Collapse" if two_collapsed_samples[i] != samples[i] else "No Op" for i in range(n)])

#         x.extend([dfa_builder(d) for d in one_step_advance_samples])
#         x_hue.extend([sampler_name]*n)
#         x_style.extend(["1-Step Advance Leading Accept" if is_accepting(one_step_advance_samples[i]) else "1-Step Advance" for i in range(n)])

#     print(sampler_name, "is done")
#     if sampler_name == "Reach-Avoid Derived" and plot_type == "scatter":
#         n = old_n

# x = np.array(x)
# x_hue = np.array(x_hue)

# x_embed = gnn(x)

# palette = sns.color_palette("Set2")

# if plot_type == "scatter":
#     for method in ["tsne", "pca"]:
#         x_proj = get_projection(x_embed, method)
#         plt.figure(figsize=(8, 8))
#         sns.scatterplot(x=x_proj[:, 0], y=x_proj[:, 1], hue=x_hue, style=x_style, palette=palette, alpha=0.5)
#         plt.xlabel("1st T-SNE Dimension")
#         plt.ylabel("2nd T-SNE Dimension")
#         plt.xticks([])
#         plt.yticks([])
#         plt.legend(ncol=3, bbox_to_anchor=(0.5, -0.16), loc='lower center')
#         plt.tight_layout()
#         plt.savefig("figs/" + gnn_type + "_" + plot_type + "_" + str(n) + "_" +  method + "_" + exp_id + ".pdf", bbox_inches='tight')
# elif plot_type == "clustermap":
#     colors = []
#     for i in range(len(sampler_names)):
#         colors.extend([palette[i]]*n)

#     colors = np.array(colors)

#     plt.figure(figsize=(8, 8))

#     cm = sns.clustermap(distance_matrix(x_embed, x_embed), row_cluster=False, col_cluster=False, row_colors=colors, col_colors=colors, xticklabels=False, yticklabels=False, cbar_pos=(.445, .83, 0.3, .02), cbar_kws={"orientation": "horizontal"}, cmap="Reds")
#     cm.ax_row_dendrogram.set_visible(False)
#     cm.ax_col_dendrogram.set_visible(False)

#     plt.tight_layout()
#     plt.savefig("figs/distance_matrix_" + gnn_type + "_" + plot_type + "_" + str(n) + "_" + exp_id + ".png", bbox_inches='tight')

#     cm = sns.clustermap(cosine_similarity(x_embed, x_embed), row_cluster=False, col_cluster=False, row_colors=colors, col_colors=colors, xticklabels=False, yticklabels=False, cbar_pos=(.445, .83, 0.3, .02), cbar_kws={"orientation": "horizontal"}, cmap="Reds_r")
#     cm.ax_row_dendrogram.set_visible(False)
#     cm.ax_col_dendrogram.set_visible(False)

#     plt.tight_layout()
#     plt.savefig("figs/cosine_similarity_" + gnn_type + "_" + plot_type + "_" + str(n) + "_" + exp_id + ".png", bbox_inches='tight')
