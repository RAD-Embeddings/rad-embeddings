import numpy as np
from dfa import DFA

from utils.utils import get_model, load_model, dfa2obs
from utils.sb3.logger_callback import LoggerCallback

class Encoder():
    def __init__(self, load_file: str):
        model = load_model(load_file)
        self.n_tokens = model.policy.features_extractor.n_tokens
        self.obs2rad = model.policy.features_extractor.obs2rad
        self.rad2token = lambda _rad: model.policy.action_net(_rad).argmax(dim=1)
        self.rad2val = model.policy.value_net

    def dfa2rad(self, dfa: DFA) -> np.array:
        assert len(dfa.inputs) == self.n_tokens
        obs = dfa2obs(dfa)
        rad = self.obs2rad(obs)
        return rad

    @staticmethod
    def train(
        env_id: str,
        save_dir: str,
        alg: str,
        reparam: bool,
        id: str = "rad",
        seed: int | None = None
        ):
        save_dir = save_dir[:-1] if save_dir.endswith("/") else save_dir
        model, config = get_model(env_id, alg, reparam, save_dir, seed)

        print("Total number of parameters:", sum(p.numel() for p in model.policy.parameters() if p.requires_grad))
        print(model.policy)

        model.learn(1_000_000, callback=LoggerCallback(gamma=config["gamma"]))
        model.save(f"{save_dir}/{id}")

