from typing import Iterable


def pv_to_A(pv: float, voltages: Iterable[float]):
    voltages = set(voltages)

    assert len(
        voltages) == 1, "Make sure that all EVSEs have the same voltage!"
    return pv / next(iter(voltages))
