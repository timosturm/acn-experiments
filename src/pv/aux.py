import numpy as np


def _save_divide(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.divide(
        a,
        b,
        # checking both for 0 prevents floating point errors
        where=(a != 0) & (b != 0),
        out=np.zeros_like(a, dtype=np.float32)
    )


def pv_utilization(energy_total, pv_total) -> float:
    """Calculates how much of the total charging energy (energy_total) was provided by PV.

    Args:
        energy_total (np.ndarray): Sum of charging rates for each timestep
        pv_total (np.ndarray): Sum of PV energy for each timestep

    Returns:
        float:
    """
    utilization = np.clip(energy_total, a_min=None, a_max=pv_total)
    utilization_ratio = _save_divide(utilization, energy_total)

    return np.mean(utilization_ratio)


def grid_use(energy_total, pv_total) -> float:
    """Calculates how much of the total charging energy (energy_total) was provided by the grid (and not by PV).

    Args:
        energy_total (np.ndarray): Sum of charging rates for each timestep
        pv_total (np.ndarray): Sum of PV energy for each timestep

    Returns:
        float:
    """
    energy_grid = np.clip(energy_total - pv_total, a_min=0, a_max=None)
    ratio = _save_divide(energy_grid, energy_total)

    return np.mean(ratio)


def unused_pv(energy_total, pv_total) -> float:
    """Calculates how much of the energy provided by PV (pv_total) was not used for charging.
    
    Args:
        energy_total (np.ndarray): Sum of charging rates for each timestep
        pv_total (np.ndarray): Sum of PV energy for each timestep

    Returns:
        float:
    """
    pv_unused = np.clip(pv_total - energy_total, a_min=0, a_max=None)
    ratio = _save_divide(pv_unused, pv_total)

    return np.mean(ratio)
