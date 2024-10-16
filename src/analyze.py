from dfa_samplers import getDFASampler, BroadcastNegation
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import utils
from gnns.graphs.GNN import GNNMaker
import numpy as np
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from scipy.spatial import distance_matrix
import seaborn as sns
import operator as OP
from sklearn.manifold import spectral_embedding
from functools import reduce
from model import ACModel
from torch_ac import DictList
import random
from dfa import DFA
import sys
from sklearn.metrics.pairwise import cosine_similarity

gnn_type = sys.argv[1]
plot_type = sys.argv[2]
exp_id = sys.argv[3]

def get_gnn(gnn_type, pretrained_model_dir):
    gnn = GNNMaker(gnn_type, len(propositions) + len(utils.feature_inds), 32)

    pretrained_status = utils.get_status(pretrained_model_dir)

    model_state = pretrained_status["model_state"]
    new_model_state = pretrained_status["model_state"].copy()

    for key in model_state.keys():
        if key.find("actor") != -1 or key.find("critic") != -1:
            del new_model_state[key]

    gnn.load_state_dict(new_model_state, strict=False)

    for param in gnn.parameters():
        param.requires_grad = False

    return gnn

def get_samples(sampler, n, gnn, dfa_builder):
    samples = [sampler.sample() for _ in range(n)]
    return gnn(np.array([dfa_builder(dfa_goal) for dfa_goal in samples])), gnn(np.array([dfa_builder(BroadcastNegation._negate(dfa_goal)) for dfa_goal in samples]))

def _to_monolithic_dfa(dfa_goal, minimize):
    if minimize:
        return ((reduce(OP.and_, map(lambda dfa_clause: reduce(OP.or_, dfa_clause), dfa_goal)).minimize(),),)
    return ((reduce(OP.and_, map(lambda dfa_clause: reduce(OP.or_, dfa_clause), dfa_goal)),),)

def get_projection(x, method):
    if method == "tsne":
        return TSNE(n_components=2, learning_rate='auto', init='random', perplexity=3).fit_transform(x)
    elif method == "pca":
        return PCA(n_components=2).fit_transform(x)
    elif method == "spectral":
        return spectral_embedding(distance_matrix(x, x), n_components=2)
    # model = make_pipeline(StandardScaler(), TSNE(n_components=3, learning_rate='auto', init='random', perplexity=3))
    # model = TSNE(n_components=3, learning_rate='auto', init='random', perplexity=3)
    # model = umap.UMAP(n_components=3)
    # return spectral_embedding(distance_matrix(x, x), n_components=2)
    return None

def collapse_conjunctions(dfa_goal, k=2):
    n = len(dfa_goal)
    if n < 2:
        return dfa_goal
    sub_goal_idx = random.sample(range(n), k)
    other_goal = []
    sub_goal = []
    for i in range(n):
        if i in sub_goal_idx:
            sub_goal.append(dfa_goal[i])
        else:
            other_goal.append(dfa_goal[i])
    other_goal = tuple(other_goal)
    sub_goal = tuple(sub_goal)
    return other_goal + _to_monolithic_dfa(sub_goal, True)

def _advance(dfa_goal, truth_assignment):
        return tuple(tuple(dfa.advance(truth_assignment).minimize() for dfa in dfa_clause) for dfa_clause in dfa_goal)

def advance_dfas(dfa_goal, k=1):
    mono = _to_monolithic_dfa(dfa_goal, False)[0][0]
    word = mono.find_word()
    return _advance(dfa_goal, word[:k])

def is_accepting(dfa_goal):
    accept = []
    for dfa_clause in dfa_goal:
        for dfa in dfa_clause:
            accept.append(dfa._label(dfa.start))
    return all(accept)


n = 1000 if plot_type == "scatter" else 100
# gnn_type = "GATv2Conv"
# gnn_type = "RGCN_8x32_ROOT_SHARED"
pretrained_model_dir = "archive/" + gnn_type + "-dumb_ac_CompositionalGeneralDFA_Simple-LTL-Env-v0_seed:1_epochs:2_bs:1024_fpp:512_dsc:0.9_lr:0.001_ent:0.01_clip:0.1_prog:full_dfa:True/train"
propositions = "abcdefghijkl"
dfa_builder = utils.DFABuilder(propositions)
gnn = get_gnn(gnn_type, pretrained_model_dir)

sampler_names = ["Reach-Avoid Derived", "Reach-Avoid", "Reach", "Reach-Avoid with Redemption", "Parity"]

x = []
x_hue = []
x_style = []
# x_size = []
for sampler_name in sampler_names:
    if sampler_name == "Reach-Avoid Derived" and plot_type == "scatter":
        old_n = n
        n *= 4
    sampler_id = sampler_name.replace(" ", "").replace("-", "").replace("with", "")
    sampler = getDFASampler("Compositional" + sampler_id + "_2_2_4_4", propositions)
    samples = [sampler.sample() for _ in range(n)]
    two_collapsed_samples = [collapse_conjunctions(dfa_goal, k=2) for dfa_goal in samples]
    one_step_advance_samples = [advance_dfas(dfa_goal, k=1) for dfa_goal in samples]

    x.extend([dfa_builder(d) for d in samples])
    x_hue.extend([sampler_name]*n)
    x_style.extend(["No Op"]*n)

    if plot_type == "scatter":

        x.extend([dfa_builder(d) for d in two_collapsed_samples])
        x_hue.extend([sampler_name]*n)
        x_style.extend(["2-Conjunction Collapse" if two_collapsed_samples[i] != samples[i] else "No Op" for i in range(n)])

        x.extend([dfa_builder(d) for d in one_step_advance_samples])
        x_hue.extend([sampler_name]*n)
        x_style.extend(["1-Step Advance Leading Accept" if is_accepting(one_step_advance_samples[i]) else "1-Step Advance" for i in range(n)])

    print(sampler_name, "is done")
    if sampler_name == "Reach-Avoid Derived" and plot_type == "scatter":
        n = old_n

x = np.array(x)
x_hue = np.array(x_hue)

x_embed = gnn(x)

palette = sns.color_palette("Set2")

if plot_type == "scatter":
    for method in ["tsne", "pca"]:
        x_proj = get_projection(x_embed, method)
        plt.figure(figsize=(8, 8))
        sns.scatterplot(x=x_proj[:, 0], y=x_proj[:, 1], hue=x_hue, style=x_style, palette=palette, alpha=0.5)
        plt.xlabel("1st T-SNE Dimension")
        plt.ylabel("2nd T-SNE Dimension")
        plt.xticks([])
        plt.yticks([])
        plt.legend(ncol=3, bbox_to_anchor=(0.5, -0.16), loc='lower center')
        plt.tight_layout()
        plt.savefig("figs/" + gnn_type + "_" + plot_type + "_" + str(n) + "_" +  method + "_" + exp_id + ".pdf", bbox_inches='tight')
elif plot_type == "clustermap":
    colors = []
    for i in range(len(sampler_names)):
        colors.extend([palette[i]]*n)

    colors = np.array(colors)

    plt.figure(figsize=(8, 8))

    cm = sns.clustermap(distance_matrix(x_embed, x_embed), row_cluster=False, col_cluster=False, row_colors=colors, col_colors=colors, xticklabels=False, yticklabels=False, cbar_pos=(.445, .83, 0.3, .02), cbar_kws={"orientation": "horizontal"}, cmap="Reds")
    cm.ax_row_dendrogram.set_visible(False)
    cm.ax_col_dendrogram.set_visible(False)

    plt.tight_layout()
    plt.savefig("figs/distance_matrix_" + gnn_type + "_" + plot_type + "_" + str(n) + "_" + exp_id + ".png", bbox_inches='tight')

    cm = sns.clustermap(cosine_similarity(x_embed, x_embed), row_cluster=False, col_cluster=False, row_colors=colors, col_colors=colors, xticklabels=False, yticklabels=False, cbar_pos=(.445, .83, 0.3, .02), cbar_kws={"orientation": "horizontal"}, cmap="Reds_r")
    cm.ax_row_dendrogram.set_visible(False)
    cm.ax_col_dendrogram.set_visible(False)

    plt.tight_layout()
    plt.savefig("figs/cosine_similarity_" + gnn_type + "_" + plot_type + "_" + str(n) + "_" + exp_id + ".png", bbox_inches='tight')

