from copy import deepcopy
import gymnasium as gym
import numpy as np
from gymportal.environment import SingleAgentSimEnv


def make_env(config, gamma, seed: int = None):
    def thunk():
        env = SingleAgentSimEnv(deepcopy(config))
        env.reset(seed=seed if seed else env.simgenerator.seed)

        env = gym.wrappers.FlattenObservation(env)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env = gym.wrappers.ClipAction(env)
        env = gym.wrappers.NormalizeObservation(env)
        env = gym.wrappers.TransformObservation(
            env, lambda obs: np.clip(obs, -10, 10))
        env = gym.wrappers.NormalizeReward(env, gamma=gamma)
        env = gym.wrappers.TransformReward(
            env, lambda reward: np.clip(reward, -10, 10))

        return env

    return thunk
