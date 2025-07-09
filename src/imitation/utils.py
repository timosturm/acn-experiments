from torch.utils.data._utils.collate import collate, default_collate_fn_map, collate_numpy_array_fn, collate_float_fn
import torch
import numpy as np

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

    # artifact = wandb.Artifact(
    #     f'model_{type}{i}',
    #     type=f'model_{type}',
    #     metadata=metadata,
    # )
    # artifact.add_file(f"{path}/{name}.mdl")
    # wandb.log_artifact(artifact)

    with open(f"{path}/{name}.json", 'w') as file:
        file.write(json.dumps(metadata))


def validate_on_env(args: EvalArgs, state_dict: OrderedDict) -> float:
    agent = args.agent_class(
        observation_shape=np.array(args.env.observation_space.shape).prod(),
        action_shape=np.array(args.env.action_space.shape).prod(),
        hiddens=args.hiddens,
    )

    agent.load_state_dict(state_dict)
    scheduler = CleanRLSchedule(agent)

    eval_sim, new_return = evaluate_model(
        scheduler,
        args.env,
        seed=args.env.unwrapped.simgenerator.seed
    )

    return eval_sim, new_return


def clean_up(args: MyArgs, run):
    
    if args.rl.envs:
        args.rl.envs.close()
        
    if args.eval.env:
        args.eval.env.close()

    run.finish()


def _collate_array(batch, *, collate_fn_map):
    return collate_numpy_array_fn(batch, collate_fn_map=collate_fn_map).to(torch.float32)


def _collate_float(batch, *, collate_fn_map):
    return collate_float_fn(batch, collate_fn_map=collate_fn_map).to(torch.float32)


def collate_to_float32(batch):
    return collate(batch, collate_fn_map=default_collate_fn_map | {np.ndarray: _collate_array, float: _collate_float})
