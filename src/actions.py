from icecream import ic
from gymportal.auxilliaries.interfaces import Interface, GymTrainedInterface
from gymportal.environment import SimReward, BaseSimInterface
from acnportal.algorithms import SortedSchedulingAlgo, first_come_first_served
from typing import List, Dict
from gymnasium import spaces
from acnportal.acnsim.interface import SessionInfo
from gymportal.environment import SimActionFactory, SimAction
from gymportal.environment.actions import _map_single_to_multi_action_space
from typing import Union

import numpy as np
from gymportal.environment import *
from typing import List, Union


def _get_min_max_rates(iface: GymTrainedInterface) -> Tuple[np.ndarray, np.ndarray]:
    """
    Gets the minimal and maximal charging rates for each station of the charging network associated with
    the iface.

    Args:
        iface: The iface that has a reference to the simulator.

    Returns:

    """

    min_rates: np.ndarray = np.array(
        [
            iface.min_pilot_signal(station_id)
            for station_id in iface.station_ids
        ]
    )
    max_rates: np.ndarray = np.array(
        [
            iface.max_pilot_signal(station_id)
            for station_id in iface.station_ids
        ]
    )

    return min_rates, max_rates


def min_max_normalization(old_min: Union[float, np.ndarray], old_max: Union[float, np.ndarray],
                          new_min: Union[float, np.ndarray], new_max: Union[float, np.ndarray],
                          values: Union[float, np.ndarray]
                          ) -> Union[np.number, np.ndarray]:
    """
    Performs min-max normalization on the given array (or value). The value is thereby transformed from the interval
    [old_min, old_max] to the new interval [new_min, new_max]. In the case that input is an array, this transformation
    is performed for each entry in the array. If one of the interval limits is an array, each entry conforms to one
    entry in input.

    Args:
        old_min: current interval minimum of the input
        old_max: current interval maximum of the input
        new_min: new interval minimum after transformation
        new_max: new interval maximum after transformation
        values: the input to be transformed

    Returns:
        input transformed to the interval [new_min, new_max]
    """

    out = new_min + ((values - old_min) *
                     (new_max - new_min) / (old_max - old_min))

    return out


def _map_single_to_multi_action_space(space: spaces.Box, iface: GymTrainedInterface) -> spaces.Dict:
    """
    Maps a Box space for single agent RL to a mapping of station ID to Box space for MARL. This is for the
    case that the space for single agent learning has the form (n_stations,).

    Args:
        space:
            The action space to convert.
        iface:
            The interface to recieve the station IDs.

    Returns:

    """
    assert space.shape == (
        len(iface.station_ids),), f"Action space must be of shape (n_stations,)={(len(iface.station_ids),)} not {space.shape}!"

    return spaces.Dict(
        {agent_id: spaces.Box(low=space.low[idx], high=space.high[idx], shape=(1,), dtype=space.dtype) for
         idx, agent_id in enumerate(iface.station_ids)})


def zero_centered_single_charging_schedule_normalized_clip(discrete: bool = False) -> SimActionFactory:
    """ Generates a SimAction instance that wraps functions to handle
    actions taking the form of a vector of pilot signals. For this
    action type, actions are assumed to be centered about 0, in that
    an action of 0 corresponds to a pilot signal of max_rate/2. So,
    to convert to a schedule, actions need to be shifted by a certain
    amount and converted to a dictionary.

    The agents actions are in the interval [-1, 1] for each station. This is transformed to the appropriate interval
    of currents [A] that the respective station can provide. E.g., if the station can provide from 6A to 32A, then -1
    corresponds to 6A and 1 corresponds to 32A.
    """

    dtype = int if discrete else np.float32

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
        action = np.clip(action, a_min=-1, a_max=1)

        min_rates, max_rates = _get_min_max_rates(iface)

        normalized_action = min_max_normalization(new_min=min_rates, new_max=max_rates, old_min=-1, old_max=1,
                                                  values=action)

        out = {
            iface.station_ids[i]: [normalized_action[i]]
            for i in range(len(normalized_action))
        }

        return out

    name = "zero-centered single schedule normalized clip"

    single = SimAction(
        space_function=space_function,
        to_schedule=to_schedule,
        name=name
    )

    multi = SimAction(
        space_function=lambda iface: _map_single_to_multi_action_space(
            space_function(iface), iface),
        to_schedule=lambda iface, action: to_schedule(iface,
                                                      np.array([action[id] for id in iface.station_ids]).flatten()),
        name=name
    )

    return SimActionFactory(single_sim_action=single, multi_sim_action=multi)


def beta_one_for_all_schedule(discrete: bool = False) -> SimActionFactory:
    # """ Generates a SimAction instance that wraps functions to handle
    # actions taking the form of a vector of pilot signals. For this
    # action type, actions are assumed to be centered about 0, in that
    # an action of 0 corresponds to a pilot signal of max_rate/2. So,
    # to convert to a schedule, actions need to be shifted by a certain
    # amount and converted to a dictionary.

    # The agents actions are in the interval [-1, 1] for each station. This is transformed to the appropriate interval
    # of currents [A] that the respective station can provide. E.g., if the station can provide from 6A to 32A, then -1
    # corresponds to 6A and 1 corresponds to 32A.
    # """

    dtype = int if discrete else np.float32

    def space_function(iface: GymTrainedInterface) -> spaces.Box:
        return spaces.Box(
            low=np.array([0], dtype=dtype),
            high=np.array([1], dtype=dtype),
            shape=(1,),
            dtype=dtype,
        )

    def to_schedule(
            iface: GymTrainedInterface, action: np.ndarray
    ) -> Dict[str, List[np.float32]]:
        min_rates, max_rates = _get_min_max_rates(iface)
        num_evses: int = len(iface.station_ids)
        space = space_function(iface)

        # Repeat the same action for each EVSE
        action = action.repeat(num_evses)
        normalized_action = min_max_normalization(
            new_min=min_rates,
            new_max=max_rates,
            old_min=space.low,
            old_max=space.high,
            values=action,
        )

        out = {
            iface.station_ids[i]: [normalized_action[i]]
            for i in range(len(normalized_action))
        }

        return out

    name = "zero-centered single schedule normalized"

    single = SimAction(
        space_function=space_function,
        to_schedule=to_schedule,
        name=name
    )

    multi = SimAction(
        space_function=lambda iface: _map_single_to_multi_action_space(
            space_function(iface), iface),
        to_schedule=lambda iface, action: to_schedule(iface,
                                                      np.array([action[id] for id in iface.station_ids]).flatten()),
        name=name
    )

    return SimActionFactory(single_sim_action=single, multi_sim_action=multi)


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
        action = {station: [0] for station in iface.station_ids} | scheduler.schedule(
            sessions)

        return action

    name = "ranking schedule"

    single = SimAction(
        space_function=space_function,
        to_schedule=to_schedule,
        name=name
    )

    multi = SimAction(
        space_function=lambda iface: _map_single_to_multi_action_space(
            space_function(iface), iface),
        to_schedule=lambda iface, action: to_schedule(iface,
                                                      np.array([action[id] for id in iface.station_ids]).flatten()),
        name=name
    )

    return SimActionFactory(single_sim_action=single, multi_sim_action=multi)


def beta_ranking_plus() -> SimActionFactory:
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
            low=np.array([0] * num_evses, dtype=dtype),
            high=np.array([1] * num_evses, dtype=dtype),
            shape=(num_evses,),
            dtype=dtype,
        )

    def to_schedule(
            iface: GymTrainedInterface, action: np.ndarray
    ) -> Dict[str, List[np.float32]]:
        def sort_fn(sessions: List[SessionInfo], iface: Interface) -> List[SessionInfo]:
            # action defines how the sessions should be sorted
            station_ids = np.array(iface._simulator.network.station_ids)

            ranking_dict = {station_id: ranking for station_id,
                            ranking in zip(station_ids, action)}

            out = sorted(
                sessions, key=lambda session: ranking_dict[session.station_id])

            return out

        scheduler = SortedSchedulingAlgo(sort_fn=sort_fn)
        scheduler.register_interface(iface)

        space = space_function(iface)
        cut_off = (space.low + space.high) * 0.2
        allowed_stations = list(np.array(iface.station_ids)[action > cut_off])
        sessions = [s for s in iface.active_sessions(
        ) if s.station_id in allowed_stations]

        action_out = {
            station: [0] for station in iface.station_ids} | scheduler.schedule(sessions)

        return action_out

    name = "ranking schedule plus"

    single = SimAction(
        space_function=space_function,
        to_schedule=to_schedule,
        name=name
    )

    multi = SimAction(
        space_function=lambda iface: _map_single_to_multi_action_space(
            space_function(iface), iface),
        to_schedule=lambda iface, action: to_schedule(iface,
                                                      np.array([action[id] for id in iface.station_ids]).flatten()),
        name=name
    )

    return SimActionFactory(single_sim_action=single, multi_sim_action=multi)


def beta_schedule_normalized(discrete: bool = False) -> SimActionFactory:
    """ Generates a SimAction instance that wraps functions to handle
    actions taking the form of a vector of pilot signals. For this
    action type, actions are assumed to be centered about 0, in that
    an action of 0 corresponds to a pilot signal of max_rate/2. So,
    to convert to a schedule, actions need to be shifted by a certain
    amount and converted to a dictionary.

    The agents actions are in the interval [-1, 1] for each station. This is transformed to the appropriate interval
    of currents [A] that the respective station can provide. E.g., if the station can provide from 6A to 32A, then -1
    corresponds to 6A and 1 corresponds to 32A.
    """

    dtype = int if discrete else np.float32

    def space_function(iface: GymTrainedInterface) -> spaces.Box:
        num_evses: int = len(iface.station_ids)

        return spaces.Box(
            low=np.array([0] * num_evses, dtype=dtype),
            high=np.array([1] * num_evses, dtype=dtype),
            shape=(num_evses,),
            dtype=dtype,
        )

    def to_schedule(
            iface: GymTrainedInterface, action: np.ndarray
    ) -> Dict[str, List[np.float32]]:
        min_rates, max_rates = _get_min_max_rates(iface)
        space = space_function(iface)

        normalized_action = min_max_normalization(
            new_min=min_rates,
            new_max=max_rates,
            old_min=space.low,
            old_max=space.high,
            values=action,
        )

        out = {
            iface.station_ids[i]: [normalized_action[i]]
            for i in range(len(normalized_action))
        }

        return out

    name = "beta schedule normalized"

    single = SimAction(
        space_function=space_function,
        to_schedule=to_schedule,
        name=name
    )

    multi = SimAction(
        space_function=lambda iface: _map_single_to_multi_action_space(
            space_function(iface), iface),
        to_schedule=lambda iface, action: to_schedule(iface,
                                                      np.array([action[id] for id in iface.station_ids]).flatten()),
        name=name
    )

    return SimActionFactory(single_sim_action=single, multi_sim_action=multi)
