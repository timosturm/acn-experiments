#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This is importent when we want to call this as a python script, because jupyter naturally has a higher recursion depth
import sys
sys.setrecursionlimit(3000)

# Print the PID when using nohup
import os
from icecream import ic
ic(os.getpid())


# In[ ]:


from gymportal.data.ev_generators import get_standard_generator, RealWorldGenerator
from acnportal.acnsim import Linear2StageBattery
from gymportal.data.battery_generators import CustomizableBatteryGenerator
from gymportal.sim import get_charging_network, Recomputer, EvaluationSimulator, SimGenerator
from datetime import datetime, timedelta

import pytz
timezone = pytz.timezone("America/Los_Angeles")


# charging_network = get_charging_network('simple_acn', basic_evse=True, voltage=208,
#                                         network_kwargs={
#                                             'station_ids': ['CA-504', 'CA-503', 'CA-502', 'CA-501'],
#                                              #'station_ids': ['CA-501'],
# "aggregate_cap": 32 * 208 / 1000})

charging_network = get_charging_network('caltech', basic_evse=True, voltage=208,
                                        network_kwargs={"transformer_cap": 150})

battery_generator = CustomizableBatteryGenerator(
    voltage=208,
    period=1,
    battery_types=[
        Linear2StageBattery],
    max_power_function="normal",
)

ev_generator = RealWorldGenerator(battery_generator=battery_generator, site='caltech', period=1)
# ev_generator = get_standard_generator(
#     'caltech', battery_generator, seed=42, frequency_multiplicator=frequency_multiplicator, duration_multiplicator=2)

train_generator = SimGenerator(
    charging_network=charging_network,
    simulation_days=7,
    n_intervals=46,
    start_date=timezone.localize(datetime(2019, 1, 1)),
    ev_generator=ev_generator,
    recomputer=Recomputer(recompute_interval=10, sparse=True),
    sim_class=EvaluationSimulator,
)

ic(train_generator.end_date + timedelta(days=1))

eval_generator = SimGenerator(
    charging_network=charging_network,
    simulation_days=7,
    n_intervals=1,
    start_date=train_generator.end_date + timedelta(days=1),
    ev_generator=ev_generator,
    recomputer=Recomputer(recompute_interval=10, sparse=True),
    sim_class=EvaluationSimulator,
)

ic(eval_generator.end_date + timedelta(days=1))

validation_generator = SimGenerator(
    charging_network=charging_network,
    simulation_days=14,
    n_intervals=1,
    start_date=eval_generator.end_date + timedelta(days=1),
    ev_generator=ev_generator,
    recomputer=Recomputer(recompute_interval=10, sparse=True),
    sim_class=EvaluationSimulator,
)

ic(validation_generator.end_date + timedelta(days=1))
pass


# In[ ]:


from src.pv.pv import read_pv_data

df_pv = read_pv_data("../pv_150kW.csv")
df_pv.describe()


# In[ ]:


from gymportal.environment import *
from src.pv.observations import pv_observation_mean
from src.pv.rewards import *

observation_objects = [
    charging_rates_observation_normalized(),
    percentage_of_magnitude_observation(),
    diff_pilots_charging_rates_observation_normalized(),
    cyclical_minute_observation(),
    cyclical_day_observation(),
    cyclical_month_observation(),
    cyclical_minute_observation_stay(),
    energy_delivered_observation_normalized(),
    num_active_stations_observation_normalized(),
    pilot_signals_observation_normalized(),
    cyclical_minute_observation_arrival(),
    cyclical_day_observation_arrival(),
    cyclical_month_observation_arrival(),
    pv_observation_mean(df_pv),
]

reward_objects = [
    # pv_utilization_reward(df_pv),
    # grid_use_penalty(df_pv),
    # unused_pv_penalty(df_pv),
    # charging_reward(),
    soft_charging_reward_pv_weighted(df_pv, transformer_cap=150),
]


# In[ ]:


import dill as pickle

with open("../caltech_#stations=54_#days=7_#intervals=46_seed=8734956.pkl", "rb") as file:
    train_generator = pickle.load(file)


# In[ ]:


steps_per_epoch = 0
for sim in train_generator._sim_memory:
    steps_per_epoch += len(sim.event_queue.queue)

ic(steps_per_epoch)


# In[ ]:


train_config = {"observation_objects": observation_objects, "action_object": zero_centered_single_charging_schedule_normalized(),
                "reward_objects": reward_objects,
                "simgenerator": train_generator,
                "meet_constraints": True}

eval_config = train_config | {'simgenerator': eval_generator}
validation_config = train_config | {'simgenerator': validation_generator}


# In[ ]:


from src.cleanRL import Args
from gymportal.evaluation import *
from src.pv.metrics import *

args = Args(
    exp_name="weighted reward + ent_coef",
    total_timesteps=steps_per_epoch * 12,
    num_steps=steps_per_epoch,
    num_envs=1,
    ent_coef=1e-4,
    # wandb:
    track=True,
    wandb_project_name="cleanRL_test",
    wandb_entity="tsturm-university-kassel",
    save_model=True,
    # my own stuff:
    train_config=train_config,
    eval_config=eval_config,
    eval_seed=930932,
    eval_metrics={
        "SoC >= 90%": percentage_soc,
        "mean SoC": mean_soc,
        "median SoC": median_soc,
        "prop feasible steps": proportion_of_feasible_charging,
        "prop feasible charge": proportion_of_feasible_charge,
        "pv utilization": lambda sim: pv_utilization_metric(sim, df_pv),
        "grid usage": lambda sim: grid_use_metric(sim, df_pv),
        "unused pv": lambda sim: unused_pv_metric(sim, df_pv),
    },
)


# In[ ]:


from src.cleanRL import train_ppo
from src.cleanRL.environment import make_env


train_ppo(args, make_env)

