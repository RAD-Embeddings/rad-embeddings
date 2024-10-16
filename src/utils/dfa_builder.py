import ring
import random
import numpy as np

import dgl
import torch
import networkx as nx
from copy import deepcopy
from pysat.solvers import Solver
from dfa import DFA

# feature_inds = {"rejecting": -1, "accepting": -2, "temp": -3, "normal": -4, "init": -5, "AND": -6, "OR": -7, "distance_normalized": -8}
# feature_inds = {"rejecting": -1, "accepting": -2, "temp": -3, "normal": -4, "init": -5, "AND": -6, "distance_normalized": -7}
feature_inds = {"rejecting": -1, "accepting": -2, "temp": -3, "normal": -4, "init": -5, "AND": -6}

"""
A class that can take an DFA formula and generate the Abstract Syntax Tree (DFA) of it. This
code can generate trees in either Networkx or DGL formats. And uses caching to remember recently
generated trees.
"""
class DFABuilder(object):
    def __init__(self, propositions):
        super(DFABuilder, self).__init__()
        self.propositions = propositions
        self.feature_size = len(self.propositions) + len(feature_inds)

    # To make the caching work.
    def __ring_key__(self):
        return "DFABuilder"

    def __call__(self, dfa_goal, library="dgl"):
        return self._to_graph(dfa_goal, library)

    def _to_graph(self, dfa_goal, library="dgl"):
        return self._to_graph_one_layer(dfa_goal, library)

    @ring.lru(maxsize=400000)
    def _to_graph_one_layer(self, dfa_goal, library="dgl"):
        from utils.env import edge_types
        nxg_goal = []
        rename_goal = []
        nxg_init_nodes = []
        for i, dfa_clause in enumerate(dfa_goal):
            for _, dfa in enumerate(dfa_clause):
                nxg, init_node = self.dfa_to_formatted_nxg(dfa)
                nxg_goal.append(nxg)
                rename_goal.append(str(i) + "_")
                nxg_init_nodes.append(str(i) + "_" + init_node)

        if nxg_goal != []:
            composed_nxg_goal = nx.union_all(nxg_goal, rename=rename_goal)
        else:
            composed_nxg_goal = nx.DiGraph()

        and_node = "AND"
        composed_nxg_goal.add_node(and_node, feat=np.array([[0.0] * self.feature_size]))
        nx.set_node_attributes(composed_nxg_goal, np.array([0.0]), "is_root")
        composed_nxg_goal.nodes[and_node]["is_root"] = np.array([1.0])
        composed_nxg_goal.nodes[and_node]["feat"][0][feature_inds["AND"]] = 1.0

        for init_node in nxg_init_nodes:
            composed_nxg_goal.add_edge(init_node, and_node, type=edge_types["AND"])

        for node in composed_nxg_goal.nodes:
            composed_nxg_goal.add_edge(node, node, type=edge_types["self"])

        nxg = composed_nxg_goal

        return self._get_dgl_graph(nxg)

    @ring.lru(maxsize=400000)
    def _to_graph_two_layers(self, dfa_goal, library="dgl"):
        from utils.env import edge_types
        nxg_goal = []
        nxg_goal_or_nodes = []
        rename_goal = []
        for i, dfa_clause in enumerate(dfa_goal):
            nxg_clause = []
            nxg_init_nodes = []
            rename_clause = []
            for j, dfa in enumerate(dfa_clause):
                nxg, init_node = self.dfa_to_formatted_nxg(dfa)
                nxg_clause.append(nxg)
                rename_clause.append(str(j) + "_")
                nxg_init_nodes.append(str(j) + "_" + init_node)

            if nxg_clause != []:
                composed_nxg_clause = nx.union_all(nxg_clause, rename=rename_clause)
                or_node = "OR"
                composed_nxg_clause.add_node(or_node, feat=np.array([[0.0] * self.feature_size]))
                composed_nxg_clause.nodes[or_node]["feat"][0][feature_inds["OR"]] = 1.0
                for nxg_init_node in nxg_init_nodes:
                    composed_nxg_clause.add_edge(nxg_init_node, or_node, type=edge_types["OR"])
                nxg_goal.append(composed_nxg_clause)
                rename_goal.append(str(i) + "_")
                nxg_goal_or_nodes.append(str(i) + "_" + or_node)

        if nxg_goal != []:
            composed_nxg_goal = nx.union_all(nxg_goal, rename=rename_goal)
        else:
            composed_nxg_goal = nx.DiGraph()

        and_node = "AND"
        composed_nxg_goal.add_node(and_node, feat=np.array([[0.0] * self.feature_size]))
        nx.set_node_attributes(composed_nxg_goal, np.array([0.0]), "is_root")
        composed_nxg_goal.nodes[and_node]["is_root"] = np.array([1.0])
        composed_nxg_goal.nodes[and_node]["feat"][0][feature_inds["AND"]] = 1.0

        for or_node in nxg_goal_or_nodes:
            composed_nxg_goal.add_edge(or_node, and_node, type=edge_types["AND"])

        for node in composed_nxg_goal.nodes:
            composed_nxg_goal.add_edge(node, node, type=edge_types["self"])

        nxg = composed_nxg_goal

        return self._get_dgl_graph(nxg)

    def _get_dgl_graph(self, nxg):

        edges = list(nxg.edges)
        nodes = list(nxg.nodes)
        edge_types_attributes = nx.get_edge_attributes(nxg, "type")

        U, V, _type = zip(*[(nodes.index(edge[0]), nodes.index(edge[1]), edge_types_attributes[edge]) for edge in edges])
        _feat, _is_root = zip(*[(nxg.nodes[node]["feat"], nxg.nodes[node]["is_root"]) for node in nodes])

        U = torch.from_numpy(np.array(U))
        V = torch.from_numpy(np.array(V))
        _type = torch.from_numpy(np.array(_type))
        _feat = torch.from_numpy(np.array(_feat))
        _is_root = torch.from_numpy(np.array(_is_root))

        g = dgl.graph((U, V))
        g.ndata["feat"] = _feat
        g.ndata["is_root"] = _is_root
        g.edata["type"] = _type

        # g.ndata["PE"] = dgl.lap_pe(g, k=2, padding=True)
        # g.ndata["PE"] = dgl.random_walk_pe(g, k=2)

        return g

    def min_distance_to_accept_by_state_normalized(self, dfa, state):
        from dfa.utils import min_distance_to_accept_by_state
        depths = min_distance_to_accept_by_state(dfa)
        if state in depths:
            return depths[state]/100.0
        return 1.0

    def _unroll_dfa_loops(self, d, k=2):
        def transition(s, c):
            s, counts = s
            counts = dict(counts)
            visit_count = counts.get(s, 0)
            if visit_count >= k:
                return (s, tuple(counts.items()))
            counts[s] = 1 + visit_count
            return (d._transition(s, c), tuple(counts.items()))
        return DFA(
                start=(d.start, ()),
                label=lambda s: d._label(s[0]),
                transition=transition,
                inputs=d.inputs
            )

    @ring.lru(maxsize=600000)
    def dfa_to_formatted_nxg(self, dfa):
        from utils.env import edge_types

        nxg = nx.DiGraph()
        new_node_name_counter = 0
        new_node_name_base_str = "temp_"

        for s in dfa.states():
            start = str(s)
            nxg.add_node(start)
            nxg.nodes[start]["feat"] = np.array([[0.0] * self.feature_size])
            nxg.nodes[start]["feat"][0][feature_inds["normal"]] = 1.0
            # Assumption: We never do more than chain length 7-8 so deviding by 100 is safe.
            # nxg.nodes[start]["feat"][0][feature_inds["distance_normalized"]] = self.min_distance_to_accept_by_state_normalized(dfa, s)
            if dfa._label(s): # is accepting?
                nxg.nodes[start]["feat"][0][feature_inds["accepting"]] = 1.0
            elif sum(s != dfa._transition(s, a) for a in dfa.inputs) == 0: # is rejecting?
                nxg.nodes[start]["feat"][0][feature_inds["rejecting"]] = 1.0
            embeddings = {}
            for a in dfa.inputs:
                e = dfa._transition(s, a)
                if s == e:
                    continue # We define self loops later when composing graphs
                end = str(e)
                if end not in embeddings.keys():
                    embeddings[end] = np.zeros(self.feature_size)
                    embeddings[end][feature_inds["temp"]] = 1.0 # Since it is a temp node
                embeddings[end][self.propositions.index(a)] = 1.0
            for end in embeddings.keys():
                new_node_name = new_node_name_base_str + str(new_node_name_counter)
                new_node_name_counter += 1
                nxg.add_node(new_node_name, feat=np.array([embeddings[end]]))
                nxg.add_edge(end, new_node_name, type=edge_types["normal-to-temp"])
                nxg.add_edge(new_node_name, start, type=edge_types["temp-to-normal"])

        init_node = str(dfa.start)
        nxg.nodes[init_node]["feat"][0][feature_inds["init"]] = 1.0

        return nxg, init_node

def draw(G, formula):
    from networkx.drawing.nx_agraph import graphviz_layout
    import matplotlib.pyplot as plt

    colors = ["black", "red", "green", "blue", "purple", "orange"]
    edge_color = [colors[i] for i in nx.get_edge_attributes(G,'type').values()]

    plt.title(formula)
    pos=graphviz_layout(G, prog='dot')
    # labels = nx.get_node_attributes(G,'token')
    labels = G.nodes
    nx.draw(G, pos, with_labels=True, arrows=True, node_shape='s', edgelist=list(nx.get_edge_attributes(G,'type')), node_size=500, node_color="white", edge_color=edge_color) #edge_color=edge_color
    plt.show()

"""
A simple test to check if the DFABuilder works fine. We do a preorder DFS traversal of the resulting
tree and convert it to a simplified formula and compare the result with the simplified version of the
original formula. They should match.
"""
if __name__ == '__main__':
    import re
    import sys
    import itertools
    import matplotlib.pyplot as plt

    sys.path.insert(0, '../../')
    from dfa_samplers import getDFASampler

    for sampler_id, _ in itertools.product(["Default", "Sequence_2_20"], range(20)):
        props = "abcdefghijklmnopqrst"
        sampler = getDFASampler(sampler_id, props)
        builder = DFABuilder(list(set(list(props))))
        formula = sampler.sample()
        tree = builder(formula, library="networkx")
        pre = list(nx.dfs_preorder_nodes(tree, source=0))
        draw(tree, formula)
        u_tree = tree.to_undirected()
        pre = list(nx.dfs_preorder_nodes(u_tree, source=0))

        original = re.sub('[,\')(]', '', str(formula))
        observed = " ".join([u_tree.nodes[i]["token"] for i in pre])

        assert original == observed, f"Test Faield: Expected: {original}, Got: {observed}"

    print("Test Passed!")
