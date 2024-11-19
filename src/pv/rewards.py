from datetime import timedelta
from typing import List, Callable, Any, Union

import numpy as np
from ray.rllib.utils.typing import MultiAgentDict

from acnportal.acnsim import Simulator
from gymportal.environment.rewards import SimReward
from gymportal.environment.interfaces import BaseSimInterface
import pandas as pd
from icecream import ic

from src.pv.utils import pv_to_A
from src.pv.pv import most_recent_P


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

        utilization_clip = np.clip(charging_sum, a_min=0, a_max=pvs_in_A)
        utilization_ratio = np.divide(
            utilization_clip,
            pvs_in_A,
            # checking both for 0 prevents floating point errors
            where=(pvs_in_A != 0) & (utilization_clip != 0),
            out=np.zeros_like(utilization_clip, dtype=float)
        )

        utilization_ratio_mean = np.mean(utilization_ratio)

        return {
            station_id: utilization_ratio_mean
            for station_id in env.interface.station_ids
        }

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
