from copy import deepcopy
import gymnasium as gym
import numpy as np
from gymportal.environment import SingleAgentSimEnv, MultiAgentSimEnv
import src.cleanRL.wrappers as wrappers


def make_env(config, gamma, i: int, seed: int = None, marl: bool = False):
    return make_env_reset(config, gamma, 0, seed, marl)


def _prepare_env(env, seed, i):
    env.reset(seed=seed if seed else env.simgenerator.seed)

    for _ in range(i):
        env.reset()


def make_env_reset(config, gamma, i: int, seed: int = None, marl: bool = False):
    def thunk():
        if marl:
            env = MultiAgentSimEnv(deepcopy(config))
            _prepare_env(env, seed, i)
            env = wrappers.MARLObservationFlatten(env)
            env = wrappers.MARLRecordEpisodeStatistics(env)
            env = wrappers.MARLClipAction(env)
            env = wrappers.MARLNormalizeReward(env)
            env = wrappers.MARLTransformReward(
                env, lambda reward: np.clip(reward, -10, 10))
        else:
            env = SingleAgentSimEnv(deepcopy(config))
            _prepare_env(env, seed, i)
            env = gym.wrappers.FlattenObservation(env)
            env = gym.wrappers.RecordEpisodeStatistics(env)
            env = gym.wrappers.ClipAction(env)
            env = gym.wrappers.NormalizeReward(env, gamma=gamma)
            env = gym.wrappers.TransformReward(
                env, lambda reward: np.clip(reward, -10, 10))

        return env

    return thunk
