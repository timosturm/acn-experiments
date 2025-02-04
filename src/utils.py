import gymnasium as gym
from acnportal.acnsim import Simulator
from itertools import tee
from typing import Optional
from gymportal.evaluation import CanSchedule

import gymnasium.spaces as spaces
from gymnasium.wrappers import FlattenObservation
from gymportal.environment import SingleAgentSimEnv
from gymportal.auxilliaries.interfaces_custom import EvaluationGymTrainingInterface


class FlattenSimEnv(FlattenObservation):

    env: SingleAgentSimEnv

    def __init__(self, config, iface_type=EvaluationGymTrainingInterface):
        self.env = SingleAgentSimEnv(config, iface_type)
        self.observation_space = spaces.flatten_space(
            self.env.observation_space)

    def observation(self, observation):
        """Flattens an observation.

        Args:
            observation: The observation to flatten

        Returns:
            The flattened observation
        """
        return spaces.flatten(self.env.observation_space, observation)


def _pairwise(iterable):
    """
    Taken from https://docs.python.org/3/library/itertools.html#itertools.pairwise, because the servers python
    version does not yet have this from itertools.

    Args:
        iterable:

    Returns:

    """
    # pairwise('ABCDEFG') --> AB BC CD DE EF FG
    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)


def evaluate_model(model: CanSchedule, eval_env: gym.Env, seed: Optional[int] = None) -> Simulator:
    """
    Evaluates a model / algorithm (either from stable_baselines3 or acnportal) by running a simulation.
    In the case of stable_baselines3 models, the predictions are made deterministically.

    Args:
        seed:
            Optional seed to make evaluations reproducible.
        env_type:
            The type of environment to use, either a single- or multi-agent environment.
        model:
            The model to produce pilot signals.
        env_config:
            Configuration dict containing rewards, actions, observations, and an interface_generating_function.
            See RebuildingEnvV2Config for details.

    Returns:
        Simulation after completion.
    """
    done = False
    observation, _ = eval_env.reset(seed=seed)
    agg_reward = 0

    while not done:

        iface = eval_env.unwrapped.interface
        action = model.get_action(observation, iface)

        observation, rew, terminated, truncated, _ = eval_env.step(
            action)
        
        agg_reward += rew

        # if isinstance(eval_env, MultiAgentEnv):
        #     done = terminated['__all__'] or truncated['__all__']
        # else:
        done = terminated or truncated

    # Get the simulator we want to return
    evaluation_simulation = eval_env.unwrapped.interface._simulator

    return evaluation_simulation, agg_reward
