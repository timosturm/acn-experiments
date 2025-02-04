import numpy as np
import torch
import torch.nn as nn
from torch.distributions.normal import Normal


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight)  # , std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(nn.Module):
    def __init__(self, observation_shape: int, action_shape: int):
        super().__init__()

        # np.array(envs.single_observation_space.shape).prod()

        self.critic = nn.Sequential(
            layer_init(
                nn.Linear(observation_shape, 128)),
            nn.Tanh(),
            layer_init(nn.Linear(128, 128)),
            nn.Tanh(),
            layer_init(nn.Linear(128, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1))  # , std=1.0),
        )

        self.actor_mean = nn.Sequential(
            layer_init(
                nn.Linear(observation_shape, 128)),
            nn.Tanh(),
            layer_init(nn.Linear(128, 128)),
            nn.Tanh(),
            layer_init(nn.Linear(128, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, action_shape))  # , std=0.01)),
        )

        self.actor_logstd = nn.Parameter(torch.zeros(1, action_shape))

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        action_mean = self.actor_mean(x)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)

        probs = Normal(action_mean, action_std)
        # if action is None:
        #   action = probs.sample()

        # reparametrisation trick; see https://arxiv.org/abs/1312.6114, Section 2.4
        # such that the action is part of the computation graph when use in loss calculation
        if action is None:
            epsilon = Normal(torch.zeros_like(action_mean),
                             torch.ones_like(action_mean)).sample()
            action = action_mean + action_std * epsilon

        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), self.critic(x)
