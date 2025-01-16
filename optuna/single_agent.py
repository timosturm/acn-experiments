#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import wandb

# This is importent when we want to call this as a python script, because jupyter naturally has a higher recursion depth
from icecream import ic
import os
import sys
sys.setrecursionlimit(3000)

# Print the PID when using nohup
ic(os.getpid())


# In[ ]:


from gymportal.data.ev_generators import get_standard_generator, RealWorldGenerator
from acnportal.acnsim import Linear2StageBattery
from gymportal.data.battery_generators import CustomizableBatteryGenerator
from gymportal.sim import get_charging_network, Recomputer, EvaluationSimulator, SimGenerator
from datetime import datetime, timedelta

import pytz
timezone = pytz.timezone("America/Los_Angeles")


charging_network = get_charging_network('simple_acn', basic_evse=True, voltage=208,
                                        network_kwargs={
                                            'station_ids': ['CA-504', 'CA-503', 'CA-502', 'CA-501'],
                                            # 'station_ids': ['CA-501'],
                                            "aggregate_cap": 32 * 208 / 1000})

# charging_network = get_charging_network('caltech', basic_evse=True, voltage=208,
#                                         network_kwargs={"transformer_cap": 150})

battery_generator = CustomizableBatteryGenerator(
    voltage=208,
    period=1,
    battery_types=[
        Linear2StageBattery],
    max_power_function="normal",
)

# ev_generator = RealWorldGenerator(battery_generator=battery_generator, site='caltech', period=1)
ev_generator = get_standard_generator(
    'caltech', battery_generator, seed=42, frequency_multiplicator=10, duration_multiplicator=2)

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


# In[4]:


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
    pv_utilization_reward(df_pv),
    grid_use_penalty(df_pv),
    unused_pv_penalty(df_pv),
    charging_reward(),
    soft_charging_reward_pv_weighted(df_pv, transformer_cap=150),
]


# In[5]:


# import dill as pickle

# with open("../caltech_#stations=54_#days=7_#intervals=46_seed=8734956.pkl", "rb") as file:
#     train_generator = pickle.load(file)


# In[ ]:


train_generator.seed = 8734956
_ = train_generator.reset()

iter = 0

while train_generator._current_date != train_generator.start_date:
    _ = train_generator.next()

    ic(iter)
    ic(train_generator._current_date)
    iter += 1


# In[ ]:


steps_per_epoch = 0
for sim in train_generator._sim_memory:
    steps_per_epoch += len(sim.event_queue.queue)

ic(steps_per_epoch)


# In[8]:


train_config = {"observation_objects": observation_objects, "action_object": zero_centered_single_charging_schedule_normalized(),
                "reward_objects": reward_objects,
                "simgenerator": train_generator,
                "meet_constraints": True}

eval_config = train_config | {'simgenerator': eval_generator}
validation_config = train_config | {'simgenerator': validation_generator}


# In[ ]:


steps_per_epoch = int(steps_per_epoch / 16)
steps_per_epoch


# In[10]:


from src.cleanRL import Args
from gymportal.evaluation import *
from src.pv.metrics import *

args = Args(
    exp_name="test on simple_acn",
    total_timesteps=steps_per_epoch * 1,
    num_steps=steps_per_epoch,
    num_envs=1,
    ent_coef=1e-4,
    # wandb:
    track=True,
    wandb_project_name="optuna_acn",
    wandb_entity="tsturm-university-kassel",
    wandb_tags=[],
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


# In[11]:


from src.cleanRL import train_ppo
from src.cleanRL.environment import make_env
from src.cleanRL.scheduler import CleanRLSchedule
from src.cleanRL.utils import load_agent
from src.utils import evaluate_model


def objective(trial, args: Args):
    args = deepcopy(args)
    args.learning_rate = trial.suggest_float(
        "learning_rate", 3e-5, 3e-3, log=True)
    args.ent_coef = trial.suggest_float("ent_coef", 1e-6, 1e-2, log=True)
    args.gamma = trial.suggest_float("gamma", 0.9, 0.99)
    args.gae_lambda = trial.suggest_float("gae_lambda", 0.95, 0.95)
    args.max_grad_norm = trial.suggest_float("max_grad_norm", 0.5, 0.5)
    args.vf_coef = trial.suggest_float("vf_coef", 0.5, 0.5)
    args.clip_coef = trial.suggest_float("clip_coef", 0.2, 0.2)

    args.wandb_tags += [
        f"ent_coef={args.ent_coef}",
        f"learning_rate={args.learning_rate}",
        f"gamma={args.gamma}",
        f"gae_lambda={args.gae_lambda}",
        f"max_grad_norm={args.max_grad_norm}",
        f"vf_coef={args.vf_coef}",
        f"clip_coef={args.clip_coef}",
    ]

    model_path = train_ppo(args, make_env)
    agent = load_agent(args, model_path)

    scheduler = CleanRLSchedule(deepcopy(agent))
    sim: EvaluationSimulator = evaluate_model(scheduler, args.eval_env,
                                              seed=args.eval_env.unwrapped.simgenerator.seed)

    # station_idx = sim.network.station_ids.index(station_id)
    timesteps = sim.reward_timesteps
    # if len(list(sim.reward_signals.values())[0].shape) > 1:
    #     stacked = np.stack([v[station_idx]
    #                        for v in sim.reward_signals.values()])
    # else:
    stacked = np.stack([v for v in sim.reward_signals.values()])

    agg_reward = np.sum(stacked)
    wandb.log({"agg_reward": agg_reward})

    return agg_reward


# In[ ]:


import optuna

study = optuna.create_study(
    study_name='distributed-example',
    storage='sqlite:///example.db',
    load_if_exists=True, 
    direction="maximize",
)

study.optimize(lambda trial: objective(trial, args), n_trials=1)


# # In[ ]:


# from src.cleanRL.utils import get_model_path, log_evaluation_plot, log_model, load_agent, log_metrics_table


# # In[ ]:


# model_path = get_model_path(args)


# # In[ ]:


# log_model(wandb.run, model_path)


# # In[ ]:


# agent = load_agent(args, model_path)


# # # In[ ]:


# from gymportal.evaluation import ACNSchedule
# from acnportal.algorithms import UncontrolledCharging, SortedSchedulingAlgo, last_come_first_served, \
#     first_come_first_served
# from src.cleanRL.scheduler import CleanRLSchedule


# models = {
#     "PPO": CleanRLSchedule(agent),
#     "FCFS": ACNSchedule(SortedSchedulingAlgo(first_come_first_served)),
#     "LCFS": ACNSchedule(SortedSchedulingAlgo(last_come_first_served)),
#     "Uncontrolled": ACNSchedule(UncontrolledCharging()),
# }


# # In[ ]:


# log_metrics_table(models, args, wandb.run)


# # In[ ]:


# log_evaluation_plot(agent, args, wandb.run)


# In[ ]:


wandb.finish()

