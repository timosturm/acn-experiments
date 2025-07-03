import acnportal
from gymportal.sim.simulators_custom import EvaluationSimulator
from datetime import timedelta
from typing import List, Callable, Any, Union

import numpy as np
from ray.rllib.utils.typing import MultiAgentDict

from acnportal.acnsim import Simulator
from gymportal.environment.rewards import SimReward
from gymportal.environment.interfaces import BaseSimInterface
import pandas as pd
from icecream import ic

from src.pv.pv import get_most_recent_P
from src.pv.utils import W_to_A

from gymportal.environment import *
import numpy as np
from src.pv.utils import get_pvs_in_W, W_to_A
import pandas as pd

from src.rewards import f
from .aux import grid_use, pv_utilization, _save_divide, unused_pv


def _energy_total(env: BaseSimInterface) -> np.ndarray:
    timestep_now = env.timestep
    timestep_prev = env.prev_timestep

    charging_rates = \
        env.interface.charging_rates[:, timestep_prev: timestep_now]

    charging_rates_sum = np.sum(charging_rates, axis=0)

    return charging_rates_sum


def _pv_total(env: BaseSimInterface, df_pv: pd.DataFrame) -> np.ndarray:
    pvs_in_A = np.array(
        [
            W_to_A(x, env.interface._simulator.network._voltages)
            for x in get_pvs_in_W(env, df_pv)
        ]
    )

    return pvs_in_A


def pv_utilization_reward(df_pv: pd.DataFrame) -> SimReward:
    """
    Rewards solar energy utilization, i.e., the more of the used charging power is from the PV, the better.
    """

    def multi_reward(env: BaseSimInterface) -> MultiAgentDict:
        value = single_reward(env)

        return {
            station_id: value for station_id in env.interface.station_ids
        }

    def single_reward(env: BaseSimInterface) -> float:
        energy_total = _energy_total(env)
        pv_total = _pv_total(env, df_pv)

        return pv_utilization(energy_total, pv_total)

    return SimReward(single_reward_function=single_reward,
                     multi_reward_function=multi_reward, name="pv_utilization")


def grid_use_penalty(df_pv: pd.DataFrame) -> SimReward:
    """
    Penalize the usage of non-pv energy.
    """

    def multi_reward(env: BaseSimInterface) -> MultiAgentDict:
        value = single_reward(env)

        return {
            station_id: value for station_id in env.interface.station_ids
        }

    def single_reward(env: BaseSimInterface) -> float:
        energy_total = _energy_total(env)
        pv_total = _pv_total(env, df_pv)

        return -grid_use(energy_total, pv_total)

    return SimReward(single_reward_function=single_reward,
                     multi_reward_function=multi_reward, name="grid_use_penalty")


def unused_pv_reward(df_pv: pd.DataFrame) -> SimReward:
    """
    Rewards usage of PV capacity for charging.
    """

    def multi_reward(env: BaseSimInterface) -> MultiAgentDict:
        value = single_reward(env)

        return {
            station_id: value for station_id in env.interface.station_ids
        }

    def single_reward(env: BaseSimInterface) -> float:
        energy_total = _energy_total(env)
        pv_total = _pv_total(env, df_pv)

        return 1 - unused_pv(energy_total, pv_total)

    return SimReward(single_reward_function=single_reward,
                     multi_reward_function=multi_reward, name="unused_pv_penalty")


def sparse_unused_pv_reward(df_pv: pd.DataFrame) -> SimReward:
    def single_reward(env: BaseSimInterface) -> float:
        evs_departing: List[acnportal.acnsim.EV] = [
            ev for ev in env.interface._simulator.ev_history.values() if ev.departure == env.timestep
        ]

        if len(evs_departing) == 0:
            return 0

        arrivals = [ev.arrival for ev in evs_departing]
        departure = env.timestep

        def get_pvs(a, b) -> np.ndarray:
            timesteps_as_dt = [
                env.interface.timestep_to_datetime(t) for t in range(a, b, env.interface._simulator.period)
            ]

            pvs_in_W = get_most_recent_P(df_pv, timesteps_as_dt)
            pvs_in_A = np.array(
                [
                    W_to_A(x, env.interface._simulator.network._voltages)
                    for x in pvs_in_W
                ]
            )

            return pvs_in_A

        pvs = [get_pvs(arrival, departure) for arrival in arrivals]

        def get_rates(a, b) -> np.ndarray:
            return np.sum(env.interface.charging_rates[:, a:b], axis=0)

        rates = [get_rates(arrival, departure) for arrival in arrivals]

        s = 0
        for p, r in zip(pvs, rates):
            v = _save_divide(p - r, p)
            s += np.mean(1 - np.clip(v, 0, None))

        s /= len(arrivals)
        return s

    return SimReward(single_reward_function=single_reward,
                     multi_reward_function=None,
                     name="sparse_unused_pv_reward")


def charging_reward() -> SimReward:
    """
    Improved normalized implementation of the soft charging reward.
    """

    def multi_reward(env: BaseSimInterface) -> MultiAgentDict:
        # Fully colaborative reward
        value = single_reward(env)
        return {
            station_id: value for station_id in env.interface.station_ids
        }

    def single_reward(env: BaseSimInterface) -> float:
        energy_total = _energy_total(env)

        sim: EvaluationSimulator = env.interface._simulator
        last_max_powers = sim.maximum_charging_power[:, env.timestep - 1]
        assert last_max_powers.shape == (len(env.interface.station_ids),)
        is_active = last_max_powers != 0

        station_dict = {
            station_id: is_active
            for station_id, is_active in zip(env.interface.station_ids, is_active)
        }

        max_pilots_sum = np.sum(
            [
                env.interface.max_pilot_signal(station_id) if is_active else 0 for station_id, is_active in station_dict.items()
            ]
        )

        ratio = _save_divide(energy_total, max_pilots_sum)

        return np.mean(ratio)

    return SimReward(single_reward_function=single_reward,
                     multi_reward_function=multi_reward, name="charging_reward")
