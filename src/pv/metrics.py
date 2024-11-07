from datetime import timedelta
from typing import Union, Callable, Iterable

from acnportal.acnsim import Simulator, ChargingNetwork
from acnportal.acnsim.analysis import *

from gymportal.environment.aux import fix_schedule_iteratively
from gymportal.sim.simulators_custom import EvaluationSimulator

from .utils import pv_to_A
from .pv import most_recent_P
import pandas as pd


def pv_utilization_mean(sim: EvaluationSimulator, df_pv: pd.DataFrame) -> float:
    """
    Measures the mean PV utilization

    Args:
        sim:

    Returns:

    """
    # timestep_now = env.timestep
    # timestep_prev = env.prev_timestep
    # sim: Simulator = env.interface._simulator

    # timesteps = np.array(
    #     list(
    #         range(timestep_prev, timestep_now, sim.period)
    #     )
    # )

    n_timesteps = sim.charging_rates.shape[1]

    timesteps_as_dt = [
        sim.start + timedelta(minutes=m) for m in range(n_timesteps)
    ]

    pvs_in_W = np.array(
        [most_recent_P(df_pv, dt) for dt in timesteps_as_dt]
    )

    pvs_in_A = [pv_to_A(x, sim.network._voltages) for x in pvs_in_W]

    charging_sum = np.sum(sim.charging_rates, axis=0)
    
    return np.mean(pvs_in_A - charging_sum)