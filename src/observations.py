from gymportal.environment.observations import SimObservationFactory, SimObservation
from gymnasium import spaces
import numpy as np
from gymportal.auxilliaries.interfaces import GymTrainedInterface
from icecream import ic

from src.actions import min_max_normalization


def minute_observation_stay() -> SimObservationFactory:
    """
    Observe the current stay duration of EVs, encoded in a cyclical manner around the minute of the day.
    """
    name = "minute_stay"

    # single = _cyclic_ev_observation(
    #     attribute_function=lambda interface, ev: extract_minute_of_day(
    #         interface.timestep_to_datetime(interface.current_time - ev.arrival)) + 1,
    #     period=minutes_per_day,
    #     name=name)

    # multi = SimObservation(
    #     lambda iface: _map_to_agent_ids(spaces.Box(
    #         low=-1, high=1, shape=(1, 2), dtype=np.float32), iface),
    #     _single_to_multi_obs(single._obs_function, same_for_all=False),
    #     name=name)

    def obs_function(iface: GymTrainedInterface) -> np.ndarray:
        attribute_values: dict = {
            station_id: 0 for station_id in iface.station_ids}
        for ev in iface.active_sessions():
            attribute_values[ev.station_id] = iface.current_time - ev.arrival

        values = np.array(list(attribute_values.values()), dtype=np.float32)

        values = min_max_normalization(
            old_min=0,
            old_max=1440,
            new_min=-1,
            new_max=1,
            values=values
        )
        return values

    single = SimObservation(
        lambda iface: spaces.Box(low=-1, high=1, shape=(len(iface.station_ids),),
                                 dtype=np.float32),
        obs_function,
        name=name)

    return SimObservationFactory(single_agent_observation=single, multi_agent_observation=None)
