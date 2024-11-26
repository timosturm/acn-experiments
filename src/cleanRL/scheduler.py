from gymportal.evaluation import CanSchedule
from typing import Union
from ray.rllib.utils.typing import MultiAgentDict

from gymnasium.core import ObsType
from gymportal.auxilliaries.interfaces import GymTrainedInterface
import torch

import torch.nn as nn


class CleanRLSchedule(CanSchedule):
    """
        Implementation of the CanSchedule interface for a torch module.
    """

    internal: nn.Module

    def __init__(self, algo: nn.Module):
        algo.eval()
        self.internal = algo

    def get_action(self, observation: Union[ObsType, MultiAgentDict], iface: GymTrainedInterface):
        # reshaping because the agent works with the SyncVectorEnv: obs (644, )-> (1, 644); action (1, 54) -> (54, )
        obs = torch.FloatTensor(observation).reshape(1, -1)
        action, _, _, _ = self.internal.get_action_and_value(obs)
        action = action.flatten()

        return action.numpy()
