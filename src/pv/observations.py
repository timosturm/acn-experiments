from datetime import timedelta
import numpy as np
from gymportal.environment.observations.auxilliaries import _single_ev_observation, _multi_ev_observation
import pandas as pd

from .utils import pv_to_A
from .pv import most_recent_P

import numpy as np
from acnportal.acnsim.interface import SessionInfo
from gymnasium import spaces

from gymportal.auxilliaries.interfaces import GymTrainedInterface
from gymportal.environment.normalization import min_max_normalization
from gymportal.environment.observations.auxilliaries import _single_ev_observation, _multi_ev_observation, \
    _single_constraints_observation, _multi_constraints_observation, _single_to_multi_obs, cyclic_transform, \
    _cyclic_ev_observation, extract_minute_of_day, _map_to_agent_ids
from gymportal.environment.observations.sim_observation import SimObservation, SimObservationFactory

from icecream import ic


def pv_observation_mean(df_pv: pd.DataFrame) -> SimObservationFactory:
    """Observation of the mean PV production in the last 5 hours in [A].

    Args:
        df_pv (pd.DataFrame): The dataframe that contains the PV data.

    Returns:
        SimObservationFactory: The observation object.
    """

    name = "pv_observation_mean"

    def single_obs_function(iface: GymTrainedInterface) -> np.ndarray:
        timesteps_as_dt = [
            iface.current_datetime - timedelta(hours=h) for h in range(5)
        ]

        pvs_in_W = np.array(
            [most_recent_P(df_pv, dt) for dt in timesteps_as_dt]
        )

        pvs_in_A = [pv_to_A(x, iface._simulator.network._voltages)
                    for x in pvs_in_W]

        return np.mean(pvs_in_A)

    single = SimObservation(
        lambda iface: spaces.Box(low=0, high=np.inf, shape=(1,),
                                 dtype=np.float32),
        single_obs_function,
        name=name)

    multi = SimObservation(
        lambda iface: _map_to_agent_ids(
            spaces.Box(low=0, high=np.inf, shape=(1, ), dtype=np.float32), iface),
        _single_to_multi_obs(single_obs_function, same_for_all=True),
        name=name)

    return SimObservationFactory(single_agent_observation=single, multi_agent_observation=multi)
