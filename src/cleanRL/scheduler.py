from copy import deepcopy
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
    """Toggle if the actions are taken stochastically or deterministic"""
    stochastic: bool 

    def __init__(self, algo: nn.Module, stochastic: bool = True):
        algo = deepcopy(algo)
        algo.eval()
        self.internal = algo
        self.stochastic = stochastic

    def get_action(self, observation: Union[ObsType, MultiAgentDict], iface: GymTrainedInterface):
        with torch.no_grad():
            # reshaping because the agent works with the SyncVectorEnv: obs (644, )-> (1, 644); action (1, 54) -> (54, )
            obs = torch.FloatTensor(observation).reshape(1, -1)
            
            if self.stochastic:
                action, _, _, _ = self.internal.get_action_and_value(obs)
            else:
                action = self.internal.actor_mean(obs)
                
            action = action.flatten()

            return action.numpy()
