from gymportal.evaluation import CanSchedule
from typing import Union
from ray.rllib.utils.typing import MultiAgentDict

from gymnasium.core import ObsType
from gymportal.auxilliaries.interfaces import GymTrainedInterface
import pytorch_lightning as pl
import torch


class CustomScheduler(CanSchedule):
    """
        Implementation of the CanSchedule interface for a pytorch-lightning model.
    """

    internal: pl.LightningModule

    def __init__(self, algo: pl.LightningModule):
        self.internal = algo

    def get_action(self, observation: Union[ObsType, MultiAgentDict], iface: GymTrainedInterface):
        obs = torch.FloatTensor(observation)
        _, action = self.internal.actor.forward(obs)
        return action.numpy()
