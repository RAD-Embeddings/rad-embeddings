"""
This class defines the environments that we are going to use.
Note that this is the place to include the right LTL-Wrapper for each environment.
"""


import gym
import gym_minigrid
import envs.gym_letters
import ltl_wrappers
import dfa_wrappers

edge_types = {}

def make_env(env_key, progression_mode, sampler, seed=None, intrinsic=0, noLTL=False, isDFAGoal=False):
    global edge_types

    env = gym.make(env_key)
    env.seed(seed)

    # Adding LTL wrappers
    if noLTL:
        return ltl_wrappers.NoLTLWrapper(env)
    elif not isDFAGoal:
        edge_types = {k:v for (v, k) in enumerate(["self", "arg", "arg1", "arg2"])}
        return ltl_wrappers.LTLEnv(env, progression_mode, sampler, intrinsic)
    else:
        edge_types = {k:v for (v, k) in enumerate(["self", "normal-to-temp", "temp-to-normal", "AND"])}
        # edge_types = {k:v for (v, k) in enumerate(["self", "normal-to-temp", "temp-to-normal", "AND", "OR"])}
        return dfa_wrappers.DFAEnv(env, progression_mode, sampler, intrinsic)
