from itertools import chain, tee
from typing import List
from gymportal.evaluation import ACNSchedule, RllibSchedule, CanSchedule
from abc import ABC, abstractmethod
from typing import Any, Union, Dict
from ray.rllib.utils.typing import MultiAgentDict

from gymnasium.core import ObsType
from gymportal.auxilliaries.interfaces import GymTrainedInterface
import pytorch_lightning as pl
import torch

import gymnasium.spaces as spaces
from gymnasium.wrappers import FlattenObservation
from gymportal.environment import SingleAgentSimEnv
from gymportal.auxilliaries.interfaces_custom import EvaluationGymTrainingInterface


class CustomSchedule(CanSchedule):
    """
        Implementation of the CanSchedule interface for a pytorch-lightning model.
    """

    internal: pl.LightningModule

    def __init__(self, algo: pl.LightningModule):
        self.internal = algo

    def get_action(self, observation: Union[ObsType, MultiAgentDict], iface: GymTrainedInterface):
        obs = torch.FloatTensor(observation)
        pi, action = self.internal.actor.forward(obs)
        return action.numpy()


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
