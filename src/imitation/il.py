from torch.utils.tensorboard import SummaryWriter


from typing import Generator, Optional, OrderedDict, Tuple

from src.cleanRL.agent import Agent

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from icecream import ic
from src.imitation.args import ImitationArgs
from src.imitation.dataset import MyDataset, TransformAction
from torch.utils.data import DataLoader
from src.imitation.utils import collate_to_float32


def imitate(
    args: ImitationArgs,
    writer: SummaryWriter,
    device: str,
    state_dict: Optional[OrderedDict],
) -> Generator[Tuple[int, OrderedDict], None, None]:

    train_ds = MyDataset(args.train_ds, transform=TransformAction())
    validation_ds = MyDataset(args.validation_ds, transform=TransformAction())

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

        loss = criterion(torch.vstack(actions), torch.vstack(true_actions))

        return loss

    val_loss = calc_val_loss(agent, validation_loader)

    for epoch in range(args.n_epochs):
        for step, x in enumerate(train_loader):
            action, _, _, _ = agent.get_action_and_value(
                x["observation"].to(device))

            loss = criterion(action, x["action"].to(device))
            writer.add_scalar("imitation/train_loss", loss.item(), epoch)

            # if step % 1 == 0:
            agent.eval()
            val_loss = calc_val_loss(agent, validation_loader)
            writer.add_scalar("imitation/validation_loss",
                                val_loss.item(), epoch)
            agent.train()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        yield epoch, agent.state_dict(), val_loss
