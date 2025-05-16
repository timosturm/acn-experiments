from acnportal.acnsim import Simulator
import pandas as pd
import numpy as np

from gymportal.environment import *
from typing import Iterable
from icecream import ic

from src.pv.pv import get_most_recent_P


def W_to_A(pv: float, voltages: Iterable[float]):
    voltages = set(voltages)

    assert len(
        voltages) == 1, "Make sure that all EVSEs have the same voltage!"
    return pv / next(iter(voltages))


def A_to_W(pv: float, voltages: Iterable[float]):
    voltages = set(voltages)

    assert len(
        voltages) == 1, "Make sure that all EVSEs have the same voltage!"
    return pv * next(iter(voltages))


def get_pvs_in_W(env: BaseSimInterface, df_pv):
    timestep_now = env.timestep - 1
    timestep_prev = env.prev_timestep
    sim: Simulator = env.interface._simulator

    timesteps_as_dt = [
        env.interface.timestep_to_datetime(t) for t in range(timestep_prev, timestep_now, sim.period)
    ]

    pvs_in_W = get_most_recent_P(df_pv, timesteps_as_dt)

    return pvs_in_W
