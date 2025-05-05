from typing import List
import acnportal
import acnportal.acnsim
import numpy as np
from gymportal.environment import SimReward, BaseSimInterface

from src.pv.aux import _save_divide


def f(x, frequency=np.pi / 2):
    """_summary_

    Args:
        x (_type_): _description_
        b (_type_, optional): _description_. Defaults to np.pi/2.

    Returns:
        _type_: _description_

    import numpy as np
    def f(x, b=np.pi / 2):
        a = 1 / np.sin(b)
        return a * np.sin(x * b)

    import matplotlib.pyplot as plt
    x_values = np.linspace(0 + 1e-5, 1, 100)

    # Plot the function for all valid b values with the additional condition f(x) >= 0
    plt.figure(figsize=(10, 6))
    for b in np.linspace(0, np.pi / 2, 100):
        plt.plot(x_values, f(x_values, b), color='green', alpha=0.1)

    plt.plot(x_values, f(x_values, np.pi/2), color="red")

    # Add labels and legend
    plt.title('Plot of f(x) = a * sin(b * x) for all valid b values with f(x) >= 0')
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.axhline(0, color='black', linewidth=0.8, linestyle='--')
    plt.axvline(0, color='black', linewidth=0.8, linestyle='--')
    plt.grid(True)
    plt.show()
    """
    amplitude = 1 / np.sin(frequency)
    return amplitude * np.sin(x * frequency)


def sparse_soc_reward() -> SimReward:
    def single_reward(env: BaseSimInterface) -> float:
        evs_departing: List[acnportal.acnsim.EV] = [
            ev for ev in env.interface._simulator.ev_history.values() if ev.departure == env.timestep
        ]

        acnportal.acnsim.Battery
        socs = [ev._battery._soc for ev in evs_departing]

        reward = _save_divide(
            np.sum([f(soc) for soc in socs]),
            np.sum([f(1) for _ in socs]),  # normalize reward
        )

        return reward

    return SimReward(single_reward_function=single_reward,
                     multi_reward_function=None,
                     name="missing_soc_penalty")


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
