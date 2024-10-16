"""
This is a simple wrapper that will include DFA goals to any given environment.
It also progress the formulas as the agent interacts with the envirionment.

However, each environment must implement the followng functions:
    - *get_events(...)*: Returns the propositions that currently hold on the environment.
    - *get_propositions(...)*: Maps the objects in the environment to a set of
                            propositions that can be referred to in DFA.

Notes about DFAEnv:
    - The episode ends if the DFA goal is progressed to True or False.
    - If the DFA goal becomes True, then an extra +1 reward is given to the agent.
    - If the DFA goal becomes False, then an extra -1 reward is given to the agent.
    - Otherwise, the agent gets the same reward given by the original environment.
"""


import numpy as np
import gym
from gym import spaces
import random
from dfa_samplers import getDFASampler
from dfa.utils import min_distance_to_accept_by_state
from functools import reduce
import operator as OP

class DFAEnv(gym.Wrapper):
    def __init__(self, env, progression_mode="full", dfa_sampler=None, intrinsic=0.0):
        """
        DFA environment
        --------------------
        It adds an DFA objective to the current environment
            - The observations become a dictionary with an added "text" field
              specifying the DFA objective
            - It also automatically progress the formula and generates an
              appropriate reward function
            - However, it does requires the user to define a labeling function
              and a set of training formulas
        progression_mode:
            - "full": the agent gets the full, progressed DFA formula as part of the observation
            - "partial": the agent sees which propositions (individually) will progress or falsify the formula
            - "none": the agent gets the full, original DFA formula as part of the observation
        """
        super().__init__(env)
        self.progression_mode = progression_mode
        self.propositions = self.env.get_propositions()
        self.sampler = getDFASampler(dfa_sampler, self.propositions)

        self.observation_space = spaces.Dict({'features': env.observation_space})
        self.intrinsic = intrinsic

        self.max_depth = 1_000_000

    def reset(self):
        self.obs = self.env.reset()

        # Defining an DFA goal
        self.dfa_goal     = self.sampler.sample()
        self.dfa_goal_original = self.dfa_goal

        # Adding the DFA goal to the observation
        if self.progression_mode == "partial":
            dfa_obs = {'features': self.obs,'progress_info': self.progress_info(self.dfa_goal)}
        else:
            dfa_obs = {'features': self.obs,'text': self.dfa_goal}
        return dfa_obs

    def step(self, action):
        # executing the action in the environment
        next_obs, original_reward, env_done, info = self.env.step(action)

        # progressing the DFA formula
        truth_assignment = self.get_events()

        old_dfa_goal = self.dfa_goal
        self.dfa_goal = self._advance(self.dfa_goal, truth_assignment)
        self.obs      = next_obs

        dfa_reward, dfa_done = self.get_dfa_reward(old_dfa_goal, self.dfa_goal)
        # dfa_reward, dfa_done = self.get_depth_reward(old_dfa_goal, self.dfa_goal)

        # Computing the new observation and returning the outcome of this action
        if self.progression_mode == "full":
            dfa_obs = {'features': self.obs,'text': self.dfa_goal}
        elif self.progression_mode == "none":
            dfa_obs = {'features': self.obs,'text': self.dfa_goal_original}
        elif self.progression_mode == "partial":
            dfa_obs = {'features': self.obs, 'progress_info': self.progress_info(self.dfa_goal)}
        else:
            raise NotImplementedError

        reward  = original_reward + dfa_reward
        done    = env_done or dfa_done

        assert dfa_reward >= -1 and dfa_reward <= 1
        assert dfa_reward !=  1 or dfa_done
        assert dfa_reward != -1 or dfa_done
        assert (dfa_reward <=  -1 or dfa_reward >= 1) or not dfa_done

        return dfa_obs, reward, done, info

    def _to_monolithic_dfa(self, dfa_goal):
        return reduce(OP.and_, map(lambda dfa_clause: reduce(OP.or_, dfa_clause), dfa_goal))

    def get_dfa_reward(self, old_dfa_goal, dfa_goal):
        if old_dfa_goal != dfa_goal:
            mono_dfa = self._to_monolithic_dfa(dfa_goal)
            if mono_dfa._label(mono_dfa.start):
                return 1.0, True
            if mono_dfa.find_word() is None:
                return -1.0, True
        return 0.0, False

    def min_distance_to_accept_by_state(self, dfa, state):
        depths = min_distance_to_accept_by_state(dfa)
        if state in depths:
            return depths[state]
        return self.max_depth

    def get_depth_reward(self, old_dfa_goal, dfa_goal):
        old_dfa = self._to_monolithic_dfa(old_dfa_goal).minimize()
        dfa = self._to_monolithic_dfa(dfa_goal).minimize()

        if dfa._label(dfa.start):
            return 1.0, True
        old_depth = self.min_distance_to_accept_by_state(old_dfa, old_dfa.start)
        depth = self.min_distance_to_accept_by_state(dfa, dfa.start)
        if depth == self.max_depth:
            return -1.0, True
        depth_reward = (old_depth - depth)/self.max_depth
        if depth_reward < 0:
            depth_reward *= 1_000
        if depth_reward > 1:
            return 1, True
        elif depth_reward < -1:
            return -1, True
        return depth_reward, False

    def _advance(self, dfa_goal, truth_assignment):
        return tuple(tuple(dfa.advance(truth_assignment).minimize() for dfa in dfa_clause) for dfa_clause in dfa_goal)

    # # X is a vector where index i is 1 if prop i progresses the formula, -1 if it falsifies it, 0 otherwise.
    def progress_info(self, dfas):
        propositions = self.env.get_propositions()
        X = np.zeros(len(self.propositions))
        for dfa in dfas:
            for i in range(len(propositions)):
                progress_i = self.dfa_progression(dfa, propositions[i])
                dfa_reward, _ = self.get_dfa_reward(progress_i)
                X[i] = dfa_reward
        return X

    def get_events(self):
        return self.env.get_events()

    def get_propositions(self):
        return self.env.get_propositions()

class NoDFAWrapper(gym.Wrapper):
    def __init__(self, env):
        """
        Removes the DFA formula from an DFAEnv
        It is useful to check the performance of off-the-shelf agents
        """
        super().__init__(env)
        self.observation_space = env.observation_space
        # self.observation_space =  env.observation_space['features']

    def reset(self):
        obs = self.env.reset()
        # obs = obs['features']
        # obs = {'features': obs}
        return obs

    def step(self, action):
        # executing the action in the environment
        obs, reward, done, info = self.env.step(action)
        # obs = obs['features']
        # obs = {'features': obs}
        return obs, reward, done, info

    def get_propositions(self):
        return list([])
