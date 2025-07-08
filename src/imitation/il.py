from typing import Generator, Optional, OrderedDict, Tuple
from src.cleanRL.agent import Agent, BetaAgent
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from src.imitation.args import ImitationArgs
from src.imitation.dataset import MyDataset, TransformAction
from torch.utils.data import DataLoader
from src.imitation.utils import collate_to_float32
from icecream import ic


def _get_min_max_action(agent_class):
    """Get the correct action space bounds for the agent class.
    That is: [-1, 1] for a gaussian output distribution and [0, 1] for the beta distribution.
    """

    if agent_class is Agent:
        return -1, 1
    elif agent_class is BetaAgent:
        return 0, 1
    else:
        raise ValueError(f"Unknown agent type: {agent_class}!")


def imitate(
    args: ImitationArgs,
    run,
    device: str,
    state_dict: Optional[OrderedDict],
) -> Generator[Tuple[int, OrderedDict], None, None]:

    min_action, max_action = _get_min_max_action(args.agent_class)
    transform = TransformAction(min_action, max_action)

    train_ds = MyDataset(args.train_ds, transform=transform)
    validation_ds = MyDataset(args.validation_ds, transform=transform)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_to_float32,
    )

    validation_loader = DataLoader(
        validation_ds,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_to_float32,
    )

    agent = args.agent_class(
        observation_shape=np.prod(train_ds[0]["observation"].shape),
        action_shape=np.prod(train_ds[0]["action"].shape),
        hiddens=args.hiddens,
    ).to(device)

    if state_dict:
        agent.load_state_dict(state_dict)

    optimizer = optim.Adam(agent.parameters(), lr=args.lr)

    criterion = nn.MSELoss()

    def calc_val_loss(agent, validation_loader):
        actions = []
        true_actions = []

        for x in validation_loader:
            action, _, _, _ = agent.get_action_and_value(
                x["observation"].to(device))

            actions.append(action)
            true_actions.append(x["action"].to(device))

        actions = torch.vstack(actions)
        true_actions = torch.vstack(true_actions)
        loss = criterion(actions, true_actions)

        return loss

    val_loss = calc_val_loss(agent, validation_loader)

    global_step = 0
    for epoch in range(args.n_epochs):
        for x in train_loader:
            global_step += 1

            action, _, _, _ = agent.get_action_and_value(
                x["observation"].to(device))

            loss = criterion(action, x["action"].to(device))

            # if step % 1 == 0:
            agent.eval()
            val_loss = calc_val_loss(agent, validation_loader)
            agent.train()

            run.log(
                {
                    "charts/epoch": global_step,
                    "imitation/train_loss": loss.item(),
                    "imitation/validation_loss": val_loss.item(),
                },
                step=global_step,
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        yield global_step, agent.state_dict(), val_loss
