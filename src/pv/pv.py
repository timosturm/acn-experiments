from datetime import datetime, timedelta
from typing import Callable, Optional

import pandas as pd


def read_pv_data(csv: str) -> pd.DataFrame:
    """
    Reads hourly PV data from https://re.jrc.ec.europa.eu/pvg_tools/en/#api_5.2

    Args:
        csv (str): path to csv file containing the PV data

    Returns:
        df (pd.DataFrame)

    """
    df = pd.read_csv(csv, skiprows=10, skipfooter=11,
                     engine="python")  # , parse_dates=["time"], date_format="%Y%m%d:%H%M")

    df.time = pd.to_datetime(df.time, format='%Y%m%d:%H%M', exact=True,
                             utc=True).map(lambda x: x.tz_convert("America/Los_Angeles"))

    return df


def most_recent_P(df: pd.DataFrame, current_time: datetime) -> float:
    """
    Returns the last available power produced for a given time. I.e., all values
    in the dataframe are assumed to apply until the next time is available.

    Args:
        df (pd.DataFrame): Pandas DataFrame
        current_time (pd.Timestamp): Current time

    Returns:
        float: Last power produced by PV
    """

    if current_time > df.time.max() + timedelta(hours=1) or current_time < df.time.min():
        raise ValueError(
            f"Requested time is out of range [{df.time.min()}, {df.time.max()}]; no data available for current_time={current_time}")

    return df.loc[df.time <= current_time].P.values[-1]

