from acnportal.acnsim import Simulator
from src.pv.pv import most_recent_P
import numpy as np

from gymportal.environment import *
from typing import Iterable


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
    timestep_now = env.timestep
    timestep_prev = env.prev_timestep
    sim: Simulator = env.interface._simulator

    timesteps = np.array(
        list(
            range(timestep_prev, timestep_now, sim.period)
        )
    )

    timesteps_as_dt = [
        env.interface.timestep_to_datetime(t) for t in timesteps
    ]

    pvs_in_W = np.array(
        [most_recent_P(df_pv, dt) for dt in timesteps_as_dt]
    )

    return pvs_in_W
