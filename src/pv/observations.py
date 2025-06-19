from datetime import timedelta
import numpy as np
import pandas as pd

from .utils import W_to_A
from .pv import get_most_recent_P

import numpy as np
from gymnasium import spaces

from gymportal.auxilliaries.interfaces import GymTrainedInterface
from gymportal.environment.normalization import min_max_normalization
from gymportal.environment.observations.auxilliaries import _single_to_multi_obs, _map_to_agent_ids
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

        pvs_in_W = get_most_recent_P(df_pv, timesteps_as_dt)

        pvs_in_A = [W_to_A(x, iface._simulator.network._voltages)
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


def pv_observation_mean_normalized(df_pv: pd.DataFrame) -> SimObservationFactory:
    """Observation of the mean PV production in the last 5 hours in [A].

    Args:
        df_pv (pd.DataFrame): The dataframe that contains the PV data.

    Returns:
        SimObservationFactory: The observation object.
    """

    name = "pv_observation_mean_normalized"

    def single_obs_function(iface: GymTrainedInterface) -> np.ndarray:
        timesteps_as_dt = [
            iface.current_datetime - timedelta(hours=h) for h in range(5)
        ]

        pvs_in_W = get_most_recent_P(df_pv, timesteps_as_dt) / df_pv.P.max()

        pvs_in_A = [W_to_A(x, iface._simulator.network._voltages)
                    for x in pvs_in_W]

        return min_max_normalization(
            old_min=0,
            old_max=W_to_A(df_pv.P.max(), iface._simulator.network._voltages),
            new_min=-1,
            new_max=1,
            values=np.mean(pvs_in_A)
        ).astype(np.float32)

    single = SimObservation(
        lambda iface: spaces.Box(low=0, high=np.inf, shape=(1,),
                                 dtype=np.float32),
        single_obs_function,
        name=name)

    multi = SimObservation(
        lambda iface: _map_to_agent_ids(
            spaces.Box(low=-1, high=1, shape=(1, ), dtype=np.float32), iface),
        _single_to_multi_obs(single_obs_function, same_for_all=True),
        name=name)

    return SimObservationFactory(single_agent_observation=single, multi_agent_observation=multi)


def pv_observation_normalized(df_pv: pd.DataFrame) -> SimObservationFactory:
    """Observation of the mean PV production in the last 5 hours in [A].

    Args:
        df_pv (pd.DataFrame): The dataframe that contains the PV data.

    Returns:
        SimObservationFactory: The observation object.
    """

    name = "pv_observation_normalized"

    def single_obs_function(iface: GymTrainedInterface) -> np.ndarray:
        timesteps_as_dt = [
            iface.current_datetime  # - timedelta(hours=h) for h in range(5)
        ]

        pvs_in_W = get_most_recent_P(df_pv, timesteps_as_dt)

        pvs_in_A = [W_to_A(x, iface._simulator.network._voltages)
                    for x in pvs_in_W]

        return min_max_normalization(
            old_min=0,
            old_max=W_to_A(df_pv.P.max(), iface._simulator.network._voltages),
            new_min=-1,
            new_max=1,
            values=np.mean(pvs_in_A)
        ).astype(np.float32)

    single = SimObservation(
        lambda iface: spaces.Box(low=0, high=np.inf, shape=(1,),
                                 dtype=np.float32),
        single_obs_function,
        name=name)

    multi = SimObservation(
        lambda iface: _map_to_agent_ids(
            spaces.Box(low=-1, high=1, shape=(1, ), dtype=np.float32), iface),
        _single_to_multi_obs(single_obs_function, same_for_all=True),
        name=name)

    return SimObservationFactory(single_agent_observation=single, multi_agent_observation=multi)
