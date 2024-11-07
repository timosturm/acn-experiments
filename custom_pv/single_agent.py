#!/usr/bin/env python
# coding: utf-8

# In[1]:


from datetime import datetime, timedelta
from tqdm import tqdm
from src.run_simulation import run_simulations, metrics


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


from src.pv.pv import read_pv_data

df_pv = read_pv_data("pv_time.csv")
df_pv.describe()


# In[ ]:


from src.pv.metrics import pv_utilization_mean
from gymportal.environment import *
from src.pv.observations import pv_observation
from src.extra import unplug_penalty, ranking_schedule
from src.pv.rewards import pv_utilization


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
    pv_observation(df_pv),
]

reward_objects = [
    # current_constraint_violation(),
    soft_charging_reward(),
    # constraint_charging_reward(),
    # unplug_penalty(),
    pv_utilization(df_pv),
    # pilot_charging_rate_difference_penalty(),
]

metrics["pv_utilization_mean"] = lambda sim: pv_utilization_mean(sim, df_pv)


# In[5]:


train_config = {"observation_objects": observation_objects, "action_object": ranking_schedule(),
                "reward_objects": reward_objects,
                "simgenerator": train_generator,
                "meet_constraints": False}

eval_config = train_config | {'simgenerator': eval_generator}
validation_config = train_config | {'simgenerator': validation_generator}


# In[6]:


env = SingleAgentSimEnv(config=eval_config)
env


# In[7]:


import matplotlib.pyplot as plt


done = False
rewards = []
obss = []
action_sum = []

obs, _ = env.reset()
while not done:
    action = env.action_space.sample()
    obs, rew, terminated, truncated, _ = env.step(action)
    done = terminated or truncated
        
    action_sum.append(np.sum(action * 16 + 32))
    obss.append(obs["pv_observation"])
    rewards.append(rew)


# In[8]:


# plt.plot(action_sum, linestyle="--", marker="x", label="charging rate")
plt.plot(rewards, linestyle=":", marker="o", label="reward")
# plt.plot(obss, linestyle="--", marker=".", label="PV production")
# plt.plot(np.array(obss) - np.array(action_sum), label="diff")
plt.legend()


# In[9]:


from acn_experiments.utils import FlattenSimEnv

from src.ppo_custom.ppo_model import PPO

env = FlattenSimEnv(config=train_config)
algo = PPO(env)


# In[10]:


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


# In[11]:


df_before = run_simulations(models, metrics=metrics, config=eval_config, seed=42)


# In[12]:


df_before.to_csv("before.csv")
ax = df_before.plot.bar()
fig = ax.get_figure()
fig.savefig("before.png", dpi=600)


# In[ ]:


import wandb
wandb.init(project="ppo_test", name="custom_pv")


# In[ ]:


from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger

wandb_logger = WandbLogger(log_model="all")

trainer = Trainer(max_epochs=24, logger=wandb_logger, accelerator="cpu", callbacks=[])
res = trainer.fit(algo)
res


# In[ ]:


wandb.finish()


# In[ ]:


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


# In[ ]:


from gymportal.plotting.plotting import plot_sim_evaluation
from gymportal.evaluation import evaluate_model

eval_sim = evaluate_model(CustomSchedule(
    algo), env_type=FlattenSimEnv, env_config=evaluation_config)


_ = plot_sim_evaluation(eval_sim)#.savefig("evaluation.png", dpi=300)


# In[ ]:


df_after = run_simulations(models, metrics=metrics, config=eval_config, seed=42)


# In[ ]:


df_after.to_csv("after.csv")
ax = df_after.plot.bar()
fig = ax.get_figure()
fig.savefig("after.png", dpi=600)


# ## Before training:

# In[ ]:


df_before


# ## After training:

# In[ ]:


df_after

