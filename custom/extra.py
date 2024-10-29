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


from gymportal.auxilliaries.interfaces import Interface, GymTrainedInterface
from gymportal.environment.actions import _map_single_to_multi_action_space
from gymportal.environment import SimActionFactory, SimAction
from acnportal.acnsim.interface import SessionInfo
from gymnasium import spaces
from typing import List, Dict
from acnportal.algorithms import SortedSchedulingAlgo


def ranking_schedule() -> SimActionFactory:
    """ Generates a SimAction instance that wraps functions to handle
    actions taking the form of a vector of pilot signals. For this
    action type, actions are assumed to be centered about 0, in that
    an action of 0 corresponds to a pilot signal of max_rate/2. So,
    to convert to a schedule, actions need to be shifted by a certain
    amount and converted to a dictionary.

    The agents actions are in the interval [-ranking, ranking] for each station. This is transformed to the appropriate interval
    of currents [A] that the respective station can provide. E.g., if the station can provide from 6A to 32A, then -ranking
    corresponds to 6A and ranking corresponds to 32A.
    """

    dtype = np.float32

    def space_function(iface: GymTrainedInterface) -> spaces.Box:
        num_evses: int = len(iface.station_ids)

        return spaces.Box(
            low=np.array([-1] * num_evses, dtype=dtype),
            high=np.array([1] * num_evses, dtype=dtype),
            shape=(num_evses,),
            dtype=dtype,
        )

    def to_schedule(
            iface: GymTrainedInterface, action: np.ndarray
    ) -> Dict[str, List[np.float32]]:
        def sort_fn(sessions: List[SessionInfo], iface: Interface) -> List[SessionInfo]:
            # action defines how the sessions should be sorted
            # s1: SessionInfo = sessions[0]

            station_ids = np.array(iface._simulator.network.station_ids)

            # sort_idx = np.argsort(action)
            # station_ids_sorted = station_ids[sort_idx]

            d = {station_id: a for station_id, a in zip(station_ids, action)}

            out = sorted(sessions, key=lambda session: d[session.station_id])

            return out

        scheduler = SortedSchedulingAlgo(sort_fn=sort_fn)

        scheduler.register_interface(iface)
        sessions = iface.active_sessions()
        action = {station: [0] for station in iface.station_ids} | scheduler.schedule(sessions)

        return action

    name = "ranking schedule"

    single = SimAction(
        space_function=space_function,
        to_schedule=to_schedule,
        name=name
    )

    multi = SimAction(
        space_function=lambda iface: _map_single_to_multi_action_space(space_function(iface), iface),
        to_schedule=lambda iface, action: to_schedule(iface,
                                                      np.array([action[id] for id in iface.station_ids]).flatten()),
        name=name
    )

    return SimActionFactory(single_sim_action=single, multi_sim_action=multi)
