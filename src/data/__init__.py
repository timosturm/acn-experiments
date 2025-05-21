import pickle
import numpy as np
import pandas as pd
import importlib.resources as pkg_resources
# from src import data
from icecream import ic


PV = "pv_150kW.csv"
DATA = "caltech_2018-03-25 00:00:00-07:53_2020-05-31 00:00:00-07:53_False.csv"
GMM = "triple_gmm+sc.pkl"


def get_data():
    path = pkg_resources.files(__package__).joinpath(DATA)

    with path.open('r', encoding='utf-8') as f:
        data = pd.read_csv(
            f,
            index_col=0,
            dtype={"kWhDelivered": np.float64, "spaceID": str},
        )

        data.connectionTime = pd.to_datetime(
            data.connectionTime,
            format='%Y-%m-%d %H:%M:%S%z',
            exact=True,
            utc=True,
        ).map(lambda x: x.tz_convert("America/Los_Angeles"))

        data.disconnectTime = pd.to_datetime(
            data.disconnectTime,
            format='%Y-%m-%d %H:%M:%S%z',
            exact=True,
            utc=True,
        ).map(lambda x: x.tz_convert("America/Los_Angeles"))

        return data


def get_gmm():
    path = pkg_resources.files(__package__).joinpath(GMM)

    with path.open('rb') as f:
        gmm, scaler = pickle.load(f)

        return gmm, scaler


def get_pv_data():
    path = pkg_resources.files(__package__).joinpath(PV)

    with path.open('r', encoding='utf-8') as f:
        df = pd.read_csv(
            f,
            skiprows=10,
            skipfooter=11,
            engine="python",
        )

        df.time = pd.to_datetime(
            df.time,
            format='%Y%m%d:%H%M',
            exact=True,
            utc=True,
        ).map(lambda x: x.tz_convert("America/Los_Angeles"))

        df = df.sort_values("time")
        return df
