import json
import wandb
from torch.utils.tensorboard import SummaryWriter


from typing import Any, Dict, Optional, OrderedDict

from src.cleanRL.scheduler import CleanRLSchedule
from src.cleanRL.agent import Agent
from src.utils import evaluate_model

import torch
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from icecream import ic
from src.imitation.args import MyArgs, EvalArgs


def save_state_dict(args, run_name: str, state_dict: OrderedDict, type: str, i: int, metadata: Optional[Dict[str, Any]] = None):
    path = f"runs/{run_name}/models/{type}/"
    name = f"{args.exp_name}_{i}"

    from pathlib import Path
    Path(path).mkdir(parents=True, exist_ok=True)

    torch.save(state_dict, f"{path}/{name}.mdl")

    artifact = wandb.Artifact(
        f'model_{type}{i}',
        type=f'model_{type}',
        metadata=metadata,
    )
    artifact.add_file(f"{path}/{name}.mdl")
    wandb.log_artifact(artifact)

    with open(f"{path}/{name}.json", 'w') as file:
        file.write(json.dumps(metadata))


def validate_on_env(args: EvalArgs, state_dict: OrderedDict) -> float:
    agent = Agent(
        observation_shape=np.array(args.env.observation_space.shape).prod(),
        action_shape=np.array(args.env.action_space.shape).prod()
    )

    agent.load_state_dict(state_dict)
    scheduler = CleanRLSchedule(agent)

    eval_sim, new_return = evaluate_model(
        scheduler,
        args.env,
        seed=args.env.unwrapped.simgenerator.seed
    )

    return eval_sim, new_return


def clean_up(args: MyArgs, writer: SummaryWriter, run):

    args.rl.envs.close()
    args.eval.env.close()

    writer.flush()
    writer.close()

    run.finish()
