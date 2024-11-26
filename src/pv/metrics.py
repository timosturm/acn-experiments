from datetime import timedelta

from acnportal.acnsim.analysis import *

from gymportal.sim.simulators_custom import EvaluationSimulator

from .utils import W_to_A
from .pv import most_recent_P
import pandas as pd

from datetime import timedelta

from acnportal.acnsim.analysis import *

from gymportal.sim.simulators_custom import EvaluationSimulator

from src.pv.utils import W_to_A
from src.pv.pv import most_recent_P
import pandas as pd
from .aux import grid_use, pv_utilization, unused_pv


def _energy_total(sim: EvaluationSimulator) -> np.ndarray:
    return np.sum(sim.charging_rates, axis=0)


def _pv_total(sim: EvaluationSimulator, df_pv: pd.DataFrame) -> np.ndarray:
    n_timesteps = sim.charging_rates.shape[1]

    timesteps_as_dt = [
        sim.start + timedelta(minutes=m) for m in range(n_timesteps)
    ]

    pvs_in_W = np.array(
        [most_recent_P(df_pv, dt) for dt in timesteps_as_dt]
    )

    pvs_in_A = np.array([W_to_A(x, sim.network._voltages) for x in pvs_in_W])

    return pvs_in_A


def pv_utilization_metric(sim: EvaluationSimulator, df_pv: pd.DataFrame) -> float:
    energy_total = _energy_total(sim)
    pv_total = _pv_total(sim, df_pv)

    return pv_utilization(energy_total, pv_total)


def grid_use_metric(sim: EvaluationSimulator, df_pv: pd.DataFrame) -> float:
    energy_total = _energy_total(sim)
    pv_total = _pv_total(sim, df_pv)

    return grid_use(energy_total, pv_total)


def unused_pv_metric(sim: EvaluationSimulator, df_pv: pd.DataFrame) -> float:
    energy_total = _energy_total(sim)
    pv_total = _pv_total(sim, df_pv)

    return unused_pv(energy_total, pv_total)


# def pv_utilization_mean(sim: EvaluationSimulator, df_pv: pd.DataFrame) -> float:
#     """
#     Measures the mean PV utilization in [%]

#     Args:
#         sim:

#     Returns:

#     """
#     n_timesteps = sim.charging_rates.shape[1]

#     timesteps_as_dt = [
#         sim.start + timedelta(minutes=m) for m in range(n_timesteps)
#     ]

#     pvs_in_W = np.array(
#         [most_recent_P(df_pv, dt) for dt in timesteps_as_dt]
#     )

#     pvs_in_A = [W_to_A(x, sim.network._voltages) for x in pvs_in_W]
#     charging_sum = np.sum(sim.charging_rates, axis=0)

#     utilization_clip = np.clip(charging_sum, a_min=0, a_max=pvs_in_A)
#     utilization_ratio = np.divide(
#         utilization_clip,
#         pvs_in_A,
#         # checking both for 0 prevents floating point errors
#         where=(pvs_in_A != 0) & (utilization_clip != 0),
#         out=np.zeros_like(utilization_clip)
#     )

#     return np.mean(utilization_ratio)
