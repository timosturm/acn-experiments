from datetime import datetime, timedelta
from icecream import ic
import pandas as pd
from gymportal.environment.interfaces import BaseSimInterface
from acnportal.acnsim import Simulator
from typing import Iterable
import json
from typing import Dict, List, Union
import gymnasium as gym
import torch
from src.cleanRL.agent import Agent
from src.cleanRL.environment import make_env
import numpy as np


def load_agent(id: str, configs: Union[List[Dict], Dict], agent_class=Agent):
    state_dict = torch.load(f"{id}.mdl", weights_only=True)
    with open(f"{id}.json", 'rb') as file:
        js = json.loads(file.read())
        hiddens = [v for k, v in js["parameter"].items() if "_layer_" in k]

    cfg = configs[0] if isinstance(configs, list) else configs

    envs = gym.vector.SyncVectorEnv(
        [make_env(cfg, 0.99, 0, 42)]
    )

    agent = agent_class(
        observation_shape=np.array(envs.single_observation_space.shape).prod(),
        action_shape=np.array(envs.single_action_space.shape).prod(),
        hiddens=hiddens,
    ).to("cpu")

    agent.load_state_dict(state_dict)
    return agent


def pv_for_sim(sim: Simulator, df_pv: pd.DataFrame) -> np.ndarray:

    pvs_in_A = np.array(
        [
            _W_to_A(x, sim.network._voltages)
            for x in _get_pvs_in_W(sim, df_pv)
        ]
    )

    return pvs_in_A


def _get_pvs_in_W(sim: Simulator, df_pv):
    timestep_now = sim.iteration - 1

    timesteps_as_dt = [
        # env.interface.timestep_to_datetime(t) for t in timesteps
        sim.start + timedelta(minutes=sim.period * i) for i in range(timestep_now)
    ]

    # Ensure both are sorted by time
    df_pv = df_pv.sort_values("time")
    df_timesteps = pd.DataFrame({"time": timesteps_as_dt}).sort_values("time")

    # Perform an asof merge
    result = pd.merge_asof(df_timesteps, df_pv,
                           on="time", direction="backward")
    pvs_in_W = result["P"].values

    return pvs_in_W


def _W_to_A(pv: float, voltages: Iterable[float]):
    voltages = set(voltages)

    assert len(
        voltages) == 1, "Make sure that all EVSEs have the same voltage!"
    return pv / next(iter(voltages))
