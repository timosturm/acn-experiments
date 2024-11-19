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

df_pv = read_pv_data("pv_time.csv")
df_pv.describe()


# In[ ]:


from src.run_simulation import metrics
from gymportal.environment import *
from src.extra import unplug_penalty
from src.pv.metrics import pv_utilization_mean
from src.pv.observations import pv_observation_mean
from src.pv.rewards import pv_utilization


def soft_charging_reward() -> SimReward:
    """
    Rewards for charge delivered in the last timestep.
    """

    def multi_reward(env: BaseSimInterface) -> MultiAgentDict:
        charging_rates = env.interface.charging_rates

        timestep = env.timestep
        prev_timestep = env.prev_timestep

        soft_reward = {
            station_id: 0 for station_id in env.interface.station_ids}

        for idx, station_id in enumerate(env.interface.station_ids):
            soft_reward[station_id] = np.sum(
                charging_rates[idx, prev_timestep: timestep]) / (env.interface.max_pilot_signal(station_id) * (
                    timestep - prev_timestep))

        return soft_reward

    def single_reward(env: BaseSimInterface) -> float:
        multi_dict = multi_reward(env)

        return float(np.sum(list(multi_dict.values()))) / len(multi_dict.keys())

    return SimReward(single_reward_function=single_reward,
                     multi_reward_function=multi_reward, name="soft_charging_reward")


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
    # current_constraint_violation(),
    # soft_charging_reward(),
    # constraint_charging_reward(),
    # unplug_penalty(),
    # pilot_charging_rate_difference_penalty(),
    pv_utilization(df_pv),
]

metrics["pv_utilization_mean"] = lambda sim: pv_utilization_mean(sim, df_pv)


# In[ ]:


from src.actions import ranking_schedule, ranking_schedule_plus, zero_centered_single_charging_schedule_normalized_clip

train_config = {"observation_objects": observation_objects, "action_object": ranking_schedule(),
                "reward_objects": reward_objects,
                "simgenerator": train_generator,
                "meet_constraints": True}

eval_config = train_config | {'simgenerator': eval_generator}
validation_config = train_config | {'simgenerator': validation_generator}


# In[ ]:


# import dill as pickle
# with open("caltech.pkl",'rb') as file:
#     env = pickle.load(file)


# In[ ]:


from src.utils import FlattenSimEnv


env = FlattenSimEnv(train_config)


# In[ ]:


lengths_load = []

for i in range(46):
    ic(f"preparing simulation {i}")
    env.reset()
    length = len(env.interface._simulator.event_queue.queue)
    lengths_load.append(length)
    
steps_per_epoch = np.sum(lengths_load) # look at all 46 weeks per epoch


# In[ ]:


from src.ppo_custom.ppo_model import PPO

algo = PPO(env, max_episode_len=np.inf, steps_per_epoch=steps_per_epoch + 1)


# In[ ]:


from gymportal.evaluation import ACNSchedule, RllibSchedule
from acnportal.algorithms import UncontrolledCharging, SortedSchedulingAlgo, last_come_first_served, \
    first_come_first_served
from src.utils import CustomSchedule


models = {
    # "PPO": CustomSchedule(algo),
    # "FCFS": ACNSchedule(SortedSchedulingAlgo(first_come_first_served)),
    # "LCFS": ACNSchedule(SortedSchedulingAlgo(last_come_first_served)),
    # "Uncontrolled": ACNSchedule(UncontrolledCharging()),
}

models


# In[ ]:


import wandb
run = wandb.init(project="ppo_x", group="PV", name=f"ranking_util_only")


# In[ ]:


from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint

from src.ppo_custom.callbacks import EvaluationMetricsCallback, EvaluationFigureCallback

wandb_logger = WandbLogger(log_model="all")
trainer = Trainer(
    max_epochs=6,
    logger=wandb_logger,
    accelerator="cpu",
    callbacks=[
        EvaluationMetricsCallback(models, metrics, eval_config, seed=42, run=run),
        ModelCheckpoint(save_top_k=-1, every_n_epochs=1,
                        save_on_train_epoch_end=True),
        EvaluationFigureCallback(charging_network, timezone, ev_generator, train_config, run=run),
    ]
)

res = trainer.fit(algo)
res


# In[ ]:


trainer.save_checkpoint("last_checkpoint.pkl")


# In[ ]:


run.finish()

