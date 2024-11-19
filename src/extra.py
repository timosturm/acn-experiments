import numpy as np
from gymportal.environment import SimReward, BaseSimInterface


def missing_soc_penalty() -> SimReward:
    def single_reward(env: BaseSimInterface) -> float:
        evs = env.interface.connected_sessions()
        demands = [ev.remaining_demand / ev.requested_energy for ev in evs]

        if len(demands) == 0:
            return 0
        else:
            return -np.sum(demands) / len(demands)

    return SimReward(single_reward_function=single_reward,
                     multi_reward_function=None,
                     name="missing_soc_penalty")


def unplug_penalty() -> SimReward:
    def single_reward(env: BaseSimInterface) -> float:
        current_time = env.interface.current_time

        evs = [ev for ev in env.interface._simulator.ev_history.values()]

        evs = [ev for ev in evs if ev.departure == current_time]

        demands = [ev.remaining_demand / ev.requested_energy for ev in evs]

        if len(demands) == 0:
            return 0
        else:
            return -np.sum(demands) / len(demands)

    return SimReward(single_reward_function=single_reward,
                     multi_reward_function=None,
                     name="unplug_penalty")
