from typing import List, Callable, Any, Union

import numpy as np
from ray.rllib.utils.typing import MultiAgentDict

from gymportal import SimReward, BaseSimInterface
from acnportal.acnsim import Simulator

def pv_utilization() -> SimReward:
    """
    If a single EVSE constraint was violated by the last schedule, a
    negative reward equal to the magnitude of the violation is added to
    the total reward.
    """

    def multi_reward(env: BaseSimInterface) -> MultiAgentDict:
        
        sim: Simulator = env.iface._simulator
        sim.
        
        # Check that each EVSE in the schedule is actually in the
        # network.
        for station_id in env.schedule:
            if station_id not in env.interface.station_ids:
                raise KeyError(
                    f"Station {station_id} in schedule but not " f"found in network."
                )

        violation = {station_id: 0 for station_id in env.interface.station_ids}

        for station_id in env.schedule:
            # Check that none of the EVSE pilot signal limits are violated.
            evse_is_continuous: bool
            evse_allowable_pilots: List[float]
            (
                evse_is_continuous,
                evse_allowable_pilots,
            ) = env.interface.allowable_pilot_signals(station_id)
            if evse_is_continuous:
                min_rate: float = evse_allowable_pilots[0]
                max_rate: float = evse_allowable_pilots[1]
                # Add penalty for any pilot signal not in
                # [min_rate, max_rate], except for 0 pilots, which aren't
                # penalized.
                violation[station_id] -= sum(
                    [
                        max(min_rate - pilot, 0) + max(pilot - max_rate, 0)
                        if pilot != 0
                        else 0
                        for pilot in env.schedule[station_id]
                    ]
                )
            else:
                # Add penalty for any pilot signal not in the list of
                # allowed pilots, except for 0 pilots, which aren't
                # penalized.
                violation[station_id] -= sum(
                    [
                        np.abs(np.array(evse_allowable_pilots) - pilot).min()
                        if pilot != 0
                        else 0
                        for pilot in env.schedule[station_id]
                    ]
                )

        return violation

    def single_reward(env: BaseSimInterface) -> float:
        multi_agent_dict = multi_reward(env)
        return float(np.sum(list(multi_agent_dict.values())))

    return SimReward(single_reward_function=single_reward,
                     multi_reward_function=multi_reward,
                     name="evse_violation")