#!/usr/bin/env python
# coding: utf-8

# In[1]:


import wandb

# This is importent when we want to call this as a python script, because jupyter naturally has a higher recursion depth
from icecream import ic
import os
import sys
sys.setrecursionlimit(3000)

# Print the PID when using nohup
ic(os.getpid())


# In[2]:


from datetime import datetime
import pickle
from typing import Optional
from gymportal.data.ev_generators import SklearnGenerator, get_data, extract_training_data
from gymportal.auxilliaries.file_utils import get_persistent_folder
import pytz
from sklearn.mixture import BayesianGaussianMixture
import numpy as np


class ScalableSklearnGenerator(SklearnGenerator):

    def __init__(
        self,
        period,
        battery_generator,
        model,
        scaler,
        frequencies_per_hour,
        duration_multiplicator=1,
        arrival_min=0,
        arrival_max=24,
        duration_min=0.0833,
        duration_max=48,
        energy_min=0.5,
        energy_max=150,
        seed=None
    ):
        super().__init__(
            period,
            battery_generator,
            model,
            frequencies_per_hour,
            duration_multiplicator,
            arrival_min, arrival_max,
            duration_min, duration_max,
            energy_min,
            energy_max,
            seed
        )

        self.scaler = scaler

    def _sample(self, n_samples: int):
        """ Generate random samples from the fitted model.

        Args:
            n_samples (int): Number of samples to generate.

        Returns:
            np.ndarray: shape (n_samples, 3), randomly generated samples. Column 1 is
                the arrival time in hours since midnight, column 2 is the session duration in hours,
                and column 3 is the energy demand in kWh.
        """
        if n_samples > 0:
            ev_matrix, _ = self.sklearn_model.sample(n_samples)
            ev_matrix = self.scaler.inverse_transform(ev_matrix)
            return self._clip_samples(ev_matrix)
        else:
            return np.array([])


# def get_generator(site, model_path: str, battery_generator, token: Optional[str] = None, seed: Optional[int] = None,
#                            frequency_multiplicator=10, duration_multiplicator=1):
#     """

#     Args:
#         site: The site which is used as a data source for the generative model.
#         battery_generator: The generator for EV batteries.
#         token: The token to access acn-data.
#         seed: A seed for random number generator
#         frequency_multiplicator: A multiplicator for the arrival frequencies of EVs, e.g., a higher value makes it
#             more likely for an EV to arrive at a given point in time.

#     Returns:

#     """
#     timezone = pytz.timezone('America/Los_Angeles')
#     data = get_data(
#         site,
#         token,
#         drop_columns=(),
#         start=datetime(2018, 3, 25, tzinfo=timezone),
#         end=datetime(2020, 5, 31, tzinfo=timezone)
#     )
#     X = extract_training_data(data)

#     try:
#         with open(model_path, "rb") as f:
#             gmm, scaler = pickle.load(f)
#     except FileNotFoundError:
#         print(f"No existing GMM found for site={site}!")

#     connection_time = X[:, 0]

#     frequencies, _ = np.histogram(connection_time, bins=range(0, 25, 1))
#     frequencies = np.array(frequencies) / np.sum(frequencies)

#     generator = ScalableSklearnGenerator(
#         period=1,
#         model=gmm,
#         scaler=scaler,
#         frequencies_per_hour=frequencies * frequency_multiplicator,
#         battery_generator=battery_generator,
#         duration_multiplicator=duration_multiplicator,
#         seed=seed
#     )

#     return generator


# In[ ]:


# from gymportal.data.ev_generators import get_standard_generator, RealWorldGenerator
from acnportal.acnsim import Linear2StageBattery
from gymportal.data.battery_generators import CustomizableBatteryGenerator
from gymportal.sim import get_charging_network, Recomputer, EvaluationSimulator, SimGenerator
from datetime import datetime, timedelta

import pytz

from src.utils import AV_pod_ids, CC_pod_ids, get_generator, get_power_function
timezone = pytz.timezone("America/Los_Angeles")

# charging_network = get_charging_network(
#     'simple_acn',
#     basic_evse=True,
#     voltage=208,
#     network_kwargs={
#         'station_ids': AV_pod_ids,
#         "aggregate_cap": (150 / 54) * len(AV_pod_ids),
#     },
# )

charging_network = get_charging_network('caltech', basic_evse=True, voltage=208,
                                        network_kwargs={"transformer_cap": 150})

battery_generator = CustomizableBatteryGenerator(
    voltage=208,
    period=1,
    battery_types=[
        Linear2StageBattery,
    ],
    max_power_function=get_power_function,
)

# ev_generator = RealWorldGenerator(battery_generator=battery_generator, site='caltech', period=1)
ev_generator = get_generator(
    'caltech',
    "../triple_gmm+sc.pkl",
    battery_generator,
    seed=42,
    frequency_multiplicator=10,
    duration_multiplicator=2,
    data="../caltech_2018-03-25 00:00:00-07:53_2020-05-31 00:00:00-07:53_False.csv",
)


# TODO Use time intervals and GMMs from https://github.com/chrisyeh96/sustaingym/blob/main/sustaingym/envs/evcharging/utils.py
# I.e., train on generated data, evaluate on new generated data and real data from the same interval
# optional: compare to "out-of-distribution" data from different interval

train_generator = SimGenerator(
    charging_network=charging_network,
    simulation_days=1,
    n_intervals=46 * 7,
    start_date=timezone.localize(datetime(2019, 1, 1)),
    ev_generator=ev_generator,
    recomputer=Recomputer(recompute_interval=10, sparse=True),
    sim_class=EvaluationSimulator,
)

ic(train_generator.end_date + timedelta(days=1))

eval_generator = SimGenerator(
    charging_network=charging_network,
    simulation_days=7 * 4,
    n_intervals=1,
    start_date=train_generator.end_date + timedelta(days=1),
    ev_generator=ev_generator,
    recomputer=Recomputer(recompute_interval=10, sparse=True),
    sim_class=EvaluationSimulator,
)

ic(eval_generator.end_date + timedelta(days=1))

# validation_generator = SimGenerator(
#     charging_network=charging_network,
#     simulation_days=14,
#     n_intervals=1,
#     start_date=eval_generator.end_date + timedelta(days=1),
#     ev_generator=ev_generator,
#     recomputer=Recomputer(recompute_interval=10, sparse=True),
#     sim_class=EvaluationSimulator,
# )

# ic(validation_generator.end_date + timedelta(days=1))
pass


# In[4]:


from src.pv.pv import read_pv_data

df_pv = read_pv_data("../pv_150kW.csv")
df_pv.describe()


# In[5]:


from gymportal.environment import *
from src.observations import minute_observation_stay
from src.pv.observations import pv_observation_mean
from src.pv.rewards import *
from src.pv.metrics import *
from gymportal.evaluation import *

from src.rewards import sparse_soc_reward


observation_objects = [
    charging_rates_observation_normalized(),
    percentage_of_magnitude_observation(),
    diff_pilots_charging_rates_observation_normalized(),
    cyclical_minute_observation(),
    cyclical_day_observation(),
    cyclical_month_observation(),
    minute_observation_stay(),
    energy_delivered_observation_normalized(),
    num_active_stations_observation_normalized(),
    pilot_signals_observation_normalized(),
    pv_observation_mean(df_pv),
]

reward_objects = [
    pv_utilization_reward(df_pv),
    unused_pv_reward(df_pv),
    sparse_soc_reward(),
]

metrics = {
    "SoC >= 90%": percentage_soc,
    "mean SoC": mean_soc,
    "median SoC": median_soc,
    "prop feasible steps": proportion_of_feasible_charging,
    "prop feasible charge": proportion_of_feasible_charge,
    "pv utilization": lambda sim: pv_utilization_metric(sim, df_pv),
    "grid usage": lambda sim: grid_use_metric(sim, df_pv),
    "unused pv": lambda sim: unused_pv_metric(sim, df_pv),
}


# In[6]:


# import dill as pickle

# with open("../caltech_#stations=54_#days=7_#intervals=46_seed=8734956.pkl", "rb") as file:
#     train_generator = pickle.load(file)


# In[7]:


train_generator.seed = 8734956 
_ = train_generator.reset()

iter = 0

while train_generator._current_date != train_generator.start_date:
    _ = train_generator.next()
    iter += 1


# In[ ]:


steps_per_epoch = 0
for eval_sim in train_generator._sim_memory:
    steps_per_epoch += len(eval_sim.event_queue.queue)

ic(steps_per_epoch)


# In[ ]:


eval_generator.seed = 8734956
_ = eval_generator.reset()

iter = 0

while eval_generator._current_date != eval_generator.start_date:
    _ = eval_generator.next()

    ic(iter)
    ic(eval_generator._current_date)
    iter += 1
    
steps_per_epoch_eval = 0
for eval_sim in eval_generator._sim_memory:
    steps_per_epoch_eval += len(eval_sim.event_queue.queue)

ic(steps_per_epoch_eval)


# In[ ]:


train_config = {"observation_objects": observation_objects, "action_object": zero_centered_single_charging_schedule_normalized(),
                "reward_objects": reward_objects,
                "simgenerator": train_generator,
                "meet_constraints": True}

eval_config = train_config | {'simgenerator': eval_generator}
# validation_config = train_config | {'simgenerator': validation_generator}


# In[ ]:


from acnportal.algorithms import UncontrolledCharging, SortedSchedulingAlgo, last_come_first_served, \
    first_come_first_served

model = ACNSchedule(SortedSchedulingAlgo(first_come_first_served))


# In[ ]:


from tqdm import tqdm
import pandas as pd

def create_df(model, name: str, env, steps_per_epoch):
    df_list = []
    
    done = False
    old_obs, _ = env.reset()

    for _ in tqdm(range(steps_per_epoch)):

        iface = env.unwrapped.interface
        action = model.get_action(old_obs, iface)

        new_obs, rew, terminated, truncated, _ = env.step(
            action)
        done = terminated or truncated

        df_list.append([old_obs.tolist(), action.tolist(), rew, done])

        if done:
            new_obs, _ = env.reset()
            done = False
            
        old_obs = new_obs
        
    df = pd.DataFrame(df_list, columns=['observation', 'action', 'reward', 'done'])
    df.to_parquet(f'{name}.parquet.gzip', compression='gzip')


# In[ ]:


from src.cleanRL.environment import make_env

create_df(model, "caltech_46_weeks_training", make_env(train_config, 0.99, 0)(), steps_per_epoch)
create_df(model, "caltech_46_weeks_validation", make_env(eval_config, 0.99, 0)(), steps_per_epoch_eval)

