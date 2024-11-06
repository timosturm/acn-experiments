from gymportal import SimObservationFactory, GymTrainedInterface, SessionInfo
import numpy as np
from gymportal.environment.observations.auxilliaries import _single_ev_observation, _multi_ev_observation

def pv_observation() -> SimObservationFactory:
    """
        Generates a SimObservation instance that wraps functions to
        observe active EV remaining energy demands in amp periods.
    """
    name = "demands"

    def attribute_function(interface: GymTrainedInterface, ev: SessionInfo) -> np.float32:
        return interface.convert_to_amp_periods(ev.remaining_demand, ev.station_id)

    single = _single_ev_observation(attribute_function, name, low=0, high=np.inf)
    multi = _multi_ev_observation(attribute_function, name, low=0, high=np.inf)

    return SimObservationFactory(single_agent_observation=single, multi_agent_observation=multi)