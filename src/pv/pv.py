from datetime import datetime, timedelta
from typing import List
import warnings
import numpy as np
import pandas as pd
from icecream import ic


def read_pv_data(csv: str) -> pd.DataFrame:
    """
    Reads hourly PV data from https://re.jrc.ec.europa.eu/pvg_tools/en/#api_5.2

    Args:
        csv (str): path to csv file containing the PV data

    Returns:
        df (pd.DataFrame)

    """
    df = pd.read_csv(csv, skiprows=10, skipfooter=11, engine="python")

    df.time = pd.to_datetime(df.time, format='%Y%m%d:%H%M', exact=True,
                             utc=True).map(lambda x: x.tz_convert("America/Los_Angeles"))

    df = df.sort_values("time")
    return df


def get_most_recent_P(df: pd.DataFrame, timesteps_as_dt: List[datetime]) -> List[float]:
    time_array = df['time'].values
    P_array = df['P'].values

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        warnings.filterwarnings("ignore", category=UserWarning)
        timesteps_as_dt = np.array(timesteps_as_dt, dtype='datetime64[ns]')

    indices = np.searchsorted(time_array, timesteps_as_dt, side='right') - 1

    if (indices < 0).any() or (indices >= len(df)).any():
        raise ValueError("Some requested times are out of data range.")

    pvs_in_W = P_array[indices]

    return pvs_in_W
