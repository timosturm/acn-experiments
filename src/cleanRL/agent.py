from typing import List
import numpy as np
import torch
import torch.nn as nn
from torch.distributions.normal import Normal
from torch.distributions.beta import Beta
from itertools import chain, tee
from typing import List
import torch.nn as nn
from icecream import ic


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


def _build_model(n_inputs: int, hiddens: List[int], n_outputs: int, activation: nn.Module, batch_norm: bool = False):
    hiddens = [n_inputs] + hiddens

    # if isinstance(activations, nn.Module):
    #     activations = [activations for _ in dropouts]

    hidden_layers = list(
        chain.from_iterable(
            (
                nn.Linear(n_in, n_out), activation, nn.BatchNorm1d(n_out)
            ) if batch_norm else (
                nn.Linear(n_in, n_out), activation
            )
            for (n_in, n_out) in _pairwise(hiddens)
        )
    )

    net = nn.Sequential(
        *hidden_layers,
        nn.Linear(hiddens[-1], n_outputs)
    )

    return net


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight)  # , std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(nn.Module):
    def __init__(self, observation_shape: int, action_shape: int, hiddens: List[int] = [128, 128, 64]):
        super().__init__()

        self.critic = _build_model(
            n_inputs=observation_shape,
            n_outputs=1,
            hiddens=hiddens,
            activation=nn.Tanh(),
        )

        self.actor_mean = _build_model(
            n_inputs=observation_shape,
            n_outputs=action_shape,
            hiddens=hiddens,
            activation=nn.Tanh(),
        )

        self.actor_logstd = nn.Parameter(torch.zeros(1, action_shape))

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        action_mean = self.actor_mean(x)

        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)

        probs = Normal(action_mean, action_std)
        if action is None:
            action = probs.rsample()

        # # reparametrisation trick; see https://arxiv.org/abs/1312.6114, Section 2.4
        # # such that the action is part of the computation graph when used in loss calculation
        # if action is None:
        #     epsilon = Normal(
        #         torch.zeros_like(action_mean),
        #         torch.ones_like(action_mean)
        #     ).sample()

        #     action = action_mean + action_std * epsilon

        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), self.critic(x)


class BetaAgent(nn.Module):
    def __init__(self, observation_shape: int, action_shape: int, hiddens: List[int] = [128, 128, 64]):
        super().__init__()

        self.critic = _build_model(
            n_inputs=observation_shape,
            n_outputs=1,
            hiddens=hiddens,
            activation=nn.Tanh(),
        )

        hiddens = np.array(hiddens)
        last_hidden = hiddens[-1]

        self._actor_base = _build_model(
            n_inputs=observation_shape,
            n_outputs=last_hidden,
            hiddens=list(hiddens[:-1]),
            activation=nn.Tanh(),
        )

        self.alpha_head = nn.Sequential(
            nn.Tanh(),
            nn.Linear(last_hidden, action_shape),
            nn.Softplus()
        )

        self.beta_head = nn.Sequential(
            nn.Tanh(),
            nn.Linear(last_hidden, action_shape),
            nn.Softplus()
        )

    def softplus(self, x):
        return torch.log(1 + torch.exp(x)) + 1

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        actor_x = self._actor_base(x)
        alpha = self.alpha_head(actor_x) + 1
        beta = self.beta_head(actor_x) + 1

        probs = Beta(alpha, beta)
        if action is None:
            action = probs.rsample()

        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), self.critic(x)


class BetaNormAgent(nn.Module):
    """Used batch normalization in addition to beta distribution for outputs + LeakyReLU as activation.
    """

    def __init__(self, observation_shape: int, action_shape: int, hiddens: List[int] = [128, 128, 64]):
        super().__init__()

        self.critic = _build_model(
            n_inputs=observation_shape,
            n_outputs=1,
            hiddens=hiddens,
            activation=nn.LeakyReLU(),
            batch_norm=True,
        )

        hiddens = np.array(hiddens)
        last_hidden = hiddens[-1]

        self._actor_base = _build_model(
            n_inputs=observation_shape,
            n_outputs=last_hidden,
            hiddens=list(hiddens[:-1]),
            activation=nn.LeakyReLU(),
            batch_norm=True,
        )

        self.alpha_head = nn.Sequential(
            nn.LeakyReLU(),
            nn.Linear(last_hidden, action_shape),
            nn.Softplus()
        )

        self.beta_head = nn.Sequential(
            nn.LeakyReLU(),
            nn.Linear(last_hidden, action_shape),
            nn.Softplus()
        )

    def softplus(self, x):
        return torch.log(1 + torch.exp(x)) + 1

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        actor_x = self._actor_base(x)
        alpha = self.alpha_head(actor_x) + 1
        beta = self.beta_head(actor_x) + 1

        probs = Beta(alpha, beta)
        if action is None:
            action = probs.rsample()

        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), self.critic(x)
