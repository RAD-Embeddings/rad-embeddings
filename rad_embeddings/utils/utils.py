import torch
import numpy as np
from dfa import DFA
from dfa.utils import dfa2dict
from stable_baselines3 import PPO
from collections import OrderedDict
from torch_geometric.data import Data
from torch_geometric.data import Batch
from dfa.utils import min_distance_to_accept_by_state

feature_inds = {"temp": -5, "rejecting": -4, "accepting": -3, "init": -2, "normal": -1}

def bisim2feat(bisim, n_tokens):
    obs1 = bisim[:, :bisim.shape[1]//2]
    obs2 = bisim[:, bisim.shape[1]//2:]
    feat1 = obs2feat(obs1, n_tokens=n_tokens)
    feat2 = obs2feat(obs2, n_tokens=n_tokens)
    return feat1, feat2

def obs2feat(dfa_obs, n_tokens):
    if dfa_obs.ndim == 1:
        return _obs2feat(dfa_obs, n_tokens=n_tokens)
    elif dfa_obs.ndim == 2:
        return Batch.from_data_list(list(map(lambda x: _obs2feat(x, n_tokens=n_tokens), dfa_obs)))
    else:
        raise ValueError(f"Invalid ndim for dfa_obs: expected 1 or 2, but got {dfa_obs.ndim}")

def _obs2feat(dfa_obs, n_tokens):
    tokens = list(range(n_tokens))
    feature_size = len(tokens) + len(feature_inds)
    dfa_int = int("".join(map(str, map(int, dfa_obs.squeeze().tolist()))))
    dfa = DFA.from_int(dfa_int, tokens)
    dfa_dict, s_init = dfa2dict(dfa)
    nodes = OrderedDict({s: np.zeros(feature_size) for s in dfa_dict.keys()})
    if len(nodes) == 1:
        edges = [(0, 0)]
    else:
        edges = [(s, s) for s in nodes]
    for s in dfa_dict.keys():
        label, transitions = dfa_dict[s]
        leaving_transitions = [1 if s != transitions[a] else 0 for a in transitions.keys()]
        if s not in nodes:
            nodes[s] = np.zeros(feature_size)
        nodes[s][feature_inds["normal"]] = 1
        if s == s_init:
            nodes[s][feature_inds["init"]] = 1
        if label: # is accepting?
            nodes[s][feature_inds["accepting"]] = 1
        elif sum(leaving_transitions) == 0: # is rejecting?
            nodes[s][feature_inds["rejecting"]] = 1
        for e in dfa_dict.keys():
            if s == e:
                continue
            for a in transitions:
                if transitions[a] == e:
                    if (s, e) not in nodes:
                        nodes[(s, e)] = np.zeros(feature_size)
                        nodes[(s, e)][feature_inds["temp"]] = 1
                    nodes[(s, e)][a] = 1
                    s_idx = list(nodes.keys()).index(s)
                    t_idx = list(nodes.keys()).index((s, e))
                    e_idx = list(nodes.keys()).index(e)
                    # Reverse
                    if (e_idx, t_idx) not in edges:
                        edges.append((e_idx, t_idx))
                    if (t_idx, t_idx) not in edges:
                        edges.append((t_idx, t_idx))
                    if (t_idx, s_idx) not in edges:
                        edges.append((t_idx, s_idx))
    feat = torch.from_numpy(np.array(list(nodes.values())))
    edge_index = torch.from_numpy(np.array(edges))
    edge_mask = torch.from_numpy(np.array([len(nodes) for _ in edges]))
    current_state = torch.from_numpy(np.array([1] + [0] * (len(nodes) - 1))) # 0 is the current state
    return Data(feat=feat, edge_index=edge_index.T, edge_mask=edge_mask, current_state=current_state)

def dfa2obs(dfa: DFA) -> Data:
    return np.array([int(i) for i in str(dfa.to_int())])

def dfa2dist(dfa_obs, n_tokens):
    tokens = list(range(n_tokens))
    feature_size = len(tokens) + len(feature_inds)

    dfa_int = int("".join(map(str, map(int, dfa_obs.squeeze().tolist()))))
    dfa = DFA.from_int(dfa_int, tokens)

    n = len(dfa.states())

    dist = min_distance_to_accept_by_state(dfa)[0]

    if dist == float("inf"):
        dist = 0

    return dist

def get_model(env_id, save_dir, alg, seed):
    from utils.config import get_config
    assert alg == "PPO" or alg == "DQN"
    config = get_config(env_id, save_dir, alg, seed)
    if alg == "DQN":
        if "Bisim" in env_id:
            from dqn import DQN
            return DQN(**config), config
        else:
            from stable_baselines3 import DQN
            return DQN(**config), config
    else:
        from stable_baselines3 import PPO
        return PPO(**config), config

def load_model(load_file):
    from stable_baselines3 import PPO
    model = PPO.load(load_file)
    model.set_parameters(load_file)
    for param in model.policy.parameters():
        param.requires_grad = False
    model.policy.eval()
    return model
