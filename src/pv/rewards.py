from datetime import timedelta
from typing import List, Callable, Any, Union

import numpy as np
from ray.rllib.utils.typing import MultiAgentDict

from acnportal.acnsim import Simulator
from gymportal.environment.rewards import SimReward
from gymportal.environment.interfaces import BaseSimInterface
import pandas as pd
from icecream import ic

from .utils import pv_to_A
from .pv import most_recent_P


def pv_utilization(df_pv: pd.DataFrame) -> SimReward:
    """Rewards the utilization of pv and penalizes using more energy than the pv produces.

    Args:
        df_pv (pd.DataFrame): A dataframe containing the pv data.

    Returns:
        SimReward: The reward object.
    """

    def multi_reward(env: BaseSimInterface) -> MultiAgentDict:

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

        pvs_in_A = [pv_to_A(x, sim.network._voltages) for x in pvs_in_W]

        charging_sum = np.sum(
            env.interface.charging_rates[:, timestep_prev: timestep_now],
            axis=0
        )

        # penalize charging_sum > pvs_in_A
        diff = np.clip(
            pvs_in_A - charging_sum, a_min=None, a_max=0
        )

        diff_sum = np.sum(diff)

        utilization = {
            station_id: 0 for station_id in env.interface.station_ids}

        for idx, station_id in enumerate(env.interface.station_ids):
            # soft_reward[station_id] = np.sum(
            # charging_rates[idx, prev_timestep: timestep]) / (env.interface.max_pilot_signal(station_id) * (
            #     timestep - prev_timestep))
            utilization[station_id] = diff_sum / (env.interface.max_pilot_signal(station_id)
                                                  * (timestep_now - timestep_prev))

        return utilization

    def single_reward(env: BaseSimInterface) -> float:
        multi_agent_dict = multi_reward(env)
        # return float(np.sum(list(multi_agent_dict.values())))
        value = 0
        for _, v in multi_agent_dict.items():
            value = v
            break

        # ic(value)

        return value

    return SimReward(single_reward_function=single_reward,
                     multi_reward_function=multi_reward,
                     name="pv_utilization")
