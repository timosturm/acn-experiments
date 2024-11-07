#!/usr/bin/env python
# coding: utf-8

# In[1]:


from datetime import datetime, timedelta
from tqdm import tqdm
from acn_experiments.run_simulation import run_simulations, metrics


# In[ ]:


from gymportal.data.ev_generators import get_standard_generator, RealWorldGenerator
from acnportal.acnsim import Linear2StageBattery
from gymportal.data.battery_generators import CustomizableBatteryGenerator
from gymportal.sim import get_charging_network, Recomputer, EvaluationSimulator, SimGenerator
from icecream import ic

import pytz
timezone = pytz.timezone("America/Los_Angeles")

# charging_network = get_charging_network('simple_acn', basic_evse=True, voltage=208,
#                                         network_kwargs={
#                                             'station_ids': ['CA-504', 'CA-503', 'CA-502', 'CA-501'],
#                                              #'station_ids': ['CA-501'],
# "aggregate_cap": 32 * 208 / 1000})

charging_network = get_charging_network('caltech', basic_evse=True, voltage=208,
                                        network_kwargs={"transformer_cap": 5 * 32 * 208 / 1000})

battery_generator = CustomizableBatteryGenerator(voltage=208,
                                                 period=1,
                                                 battery_types=[
                                                     Linear2StageBattery],
                                                 max_power_function='normal')

ev_generator = RealWorldGenerator(battery_generator=battery_generator, site='caltech', period=1)
# ev_generator = get_standard_generator('caltech', battery_generator, seed=42)

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


# In[3]:


from gymportal.environment import *
from src.extra import unplug_penalty, ranking_schedule


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
]

reward_objects = [
    # current_constraint_violation(),
    # soft_charging_reward(),
    constraint_charging_reward(),
    unplug_penalty(),
    # pilot_charging_rate_difference_penalty(),
]


# In[4]:


train_config = {"observation_objects": observation_objects, "action_object": ranking_schedule(),
                "reward_objects": reward_objects,
                "simgenerator": train_generator,
                "meet_constraints": False}

eval_config = train_config | {'simgenerator': eval_generator}
validation_config = train_config | {'simgenerator': validation_generator}


# In[5]:


from acn_experiments.utils import FlattenSimEnv
from src.ppo_custom.ppo_model import PPO

env = FlattenSimEnv(config=train_config)
algo = PPO(env)


# In[6]:


from gymportal.evaluation import ACNSchedule, RllibSchedule
from acnportal.algorithms import UncontrolledCharging, SortedSchedulingAlgo, last_come_first_served, \
    first_come_first_served
from acn_experiments.utils import CustomSchedule


models = {
    "PPO": CustomSchedule(algo),
    "FCFS": ACNSchedule(SortedSchedulingAlgo(first_come_first_served)),
    "LCFS": ACNSchedule(SortedSchedulingAlgo(last_come_first_served)),
    "Uncontrolled": ACNSchedule(UncontrolledCharging()),
}

models


# In[7]:


df_before = run_simulations(models, metrics=metrics, config=eval_config, seed=42)


# In[8]:


df_before.to_csv("before.csv")
ax = df_before.plot.bar()
fig = ax.get_figure()
fig.savefig("before.png", dpi=600)


# In[ ]:


import wandb
wandb.init(project="ppo_test", name="ranking")


# In[10]:


from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger

wandb_logger = WandbLogger(log_model="all")

# model = PPO(env)
trainer = Trainer(max_epochs=24, logger=wandb_logger, accelerator="cpu", callbacks=[])
res = trainer.fit(algo)
res


# In[11]:


wandb.finish()


# In[12]:


from datetime import datetime

eval_generator = SimGenerator(
    charging_network=charging_network,
    simulation_days=1,
    n_intervals=1,
    start_date=timezone.localize(datetime(2019, 9, 23)),
    ev_generator=ev_generator,
    recomputer=Recomputer(recompute_interval=10, sparse=True),
    sim_class=EvaluationSimulator,
)

evaluation_config = train_config | {"simgenerator": eval_generator}


# In[13]:


from gymportal.plotting.plotting import plot_sim_evaluation
from gymportal.evaluation import evaluate_model

eval_sim = evaluate_model(CustomSchedule(
    algo), env_type=FlattenSimEnv, env_config=evaluation_config)


_ = plot_sim_evaluation(eval_sim)#.savefig("evaluation.png", dpi=300)


# In[14]:


from gymportal.plotting.plotting import plot_sim_evaluation

_ = plot_sim_evaluation(eval_sim)


# In[15]:


df_after = run_simulations(models, metrics=metrics, config=eval_config, seed=42)


# In[16]:


df_after.to_csv("after.csv")
ax = df_after.plot.bar()
fig = ax.get_figure()
fig.savefig("after.png", dpi=600)


# ## Before training:

# In[17]:


df_before


# ## After training:

# In[18]:


df_after

