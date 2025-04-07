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

from src.pv.utils import W_to_A
from src.pv.pv import most_recent_P

from gymportal.environment import *
import numpy as np
from src.pv.utils import get_pvs_in_W, W_to_A
import pandas as pd
from .aux import grid_use, pv_utilization, _save_divide, unused_pv


def _energy_total(env: BaseSimInterface) -> np.ndarray:
    timestep_now = env.timestep
    timestep_prev = env.prev_timestep

    charging_rates = env.interface.charging_rates[:,
                                                  timestep_prev: timestep_now]

    charging_rates_sum = np.sum(charging_rates, axis=0)

    return charging_rates_sum


def _pv_total(env: BaseSimInterface, df_pv: pd.DataFrame) -> np.ndarray:
    pvs_in_A = np.array([W_to_A(x, env.interface._simulator.network._voltages)
                        for x in get_pvs_in_W(env, df_pv)])

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
        was_active = last_max_powers != 0

        station_dict = {station_id: is_active
                        for station_id, is_active in zip(env.interface.station_ids, was_active)}

        max_pilots_sum = np.sum(
            [
                env.interface.max_pilot_signal(station_id) if is_active else 0 for station_id, is_active in station_dict.items()
            ]
        )

        ratio = _save_divide(energy_total, max_pilots_sum)

        return np.mean(ratio)

    return SimReward(single_reward_function=single_reward,
                     multi_reward_function=multi_reward, name="charging_reward")


def soft_charging_reward_pv_weighted(df_pv: pd.DataFrame, transformer_cap: float) -> SimReward:
    """
    Rewards for charge delivered in the last timestep weighted by the percentage of available PV.
    """

    def multi_reward(env: BaseSimInterface) -> MultiAgentDict:
        raise NotImplementedError()

    def single_reward(env: BaseSimInterface) -> float:
        pvs_in_W = get_pvs_in_W(env, df_pv)
        ratio = pvs_in_W / transformer_cap

        energy_total = _energy_total(env)

        return np.mean(ratio * energy_total)

    return SimReward(single_reward_function=single_reward,
                     multi_reward_function=multi_reward, name="soft_charging_reward_pv_weighted")


# def pv_utilization_old(df_pv: pd.DataFrame) -> SimReward:
#     """Rewards the utilization of pv and penalizes using more energy than the pv produces.

#     Args:
#         df_pv (pd.DataFrame): A dataframe containing the pv data.

#     Returns:
#         SimReward: The reward object.
#     """

#     def multi_reward(env: BaseSimInterface) -> MultiAgentDict:

#         timestep_now = env.timestep
#         timestep_prev = env.prev_timestep
#         sim: Simulator = env.interface._simulator

#         timesteps = np.array(
#             list(
#                 range(timestep_prev, timestep_now, sim.period)
#             )
#         )

#         timesteps_as_dt = [
#             env.interface.timestep_to_datetime(t) for t in timesteps
#         ]

#         pvs_in_W = np.array(
#             [most_recent_P(df_pv, dt) for dt in timesteps_as_dt]
#         )

#         pvs_in_A = [W_to_A(x, sim.network._voltages) for x in pvs_in_W]

#         charging_sum = np.sum(
#             env.interface.charging_rates[:, timestep_prev: timestep_now],
#             axis=0
#         )

#         utilization_clip = np.clip(charging_sum, a_min=0, a_max=pvs_in_A)
#         utilization_ratio = np.divide(
#             utilization_clip,
#             pvs_in_A,
#             # checking both for 0 prevents floating point errors
#             where=(pvs_in_A != 0) & (utilization_clip != 0),
#             out=np.zeros_like(utilization_clip, dtype=float)
#         )

#         utilization_ratio_mean = np.mean(utilization_ratio)

#         return {
#             station_id: utilization_ratio_mean
#             for station_id in env.interface.station_ids
#         }

#     def single_reward(env: BaseSimInterface) -> float:
#         multi_agent_dict = multi_reward(env)
#         # return float(np.sum(list(multi_agent_dict.values())))
#         value = 0
#         for _, v in multi_agent_dict.items():
#             value = v
#             break

#         # ic(value)

#         return value

#     return SimReward(single_reward_function=single_reward,
#                      multi_reward_function=multi_reward,
#                      name="pv_utilization")
