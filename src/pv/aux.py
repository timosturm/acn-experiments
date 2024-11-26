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
    utilization = np.clip(energy_total, a_min=None, a_max=pv_total)
    utilization_ratio = _save_divide(utilization, energy_total)

    return np.mean(utilization_ratio)


def grid_use(energy_total, pv_total) -> float:
    energy_grid = np.clip(energy_total - pv_total, a_min=0, a_max=None)
    ratio = _save_divide(energy_grid, energy_total)

    return np.mean(ratio)


def unused_pv(energy_total, pv_total) -> float:
    pv_unused = np.clip(pv_total - energy_total, a_min=0, a_max=None)

    ratio = _save_divide(pv_unused, pv_total)

    return np.mean(ratio)
