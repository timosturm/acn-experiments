#!/usr/bin/env python
# coding: utf-8

# In[1]:


# This is importent when we want to call this as a python script, because jupyter naturally has a higher recursion depth
from datetime import datetime
import tyro
from src.pv.rewards import *
from src.pv.observations import pv_observation_mean
from gymportal.environment import *
from itertools import product
import pytz
from datetime import datetime, timedelta
from gymportal.sim import get_charging_network, Recomputer, EvaluationSimulator, SimGenerator
from gymportal.data.battery_generators import CustomizableBatteryGenerator
from acnportal.acnsim import Linear2StageBattery
from gymportal.data.ev_generators import get_standard_generator, RealWorldGenerator
from icecream import ic
import os
import sys
from src.cleanRL import Args
from gymportal.evaluation import *
from src.pv.metrics import *
from src.utils import _pairwise
sys.setrecursionlimit(3000)
# Print the PID when using nohup
ic(os.getpid())


# In[2]:

def main(transformer_cap: int, frequency_multiplicator: float, duration_multiplicator: float):

    timezone = pytz.timezone("America/Los_Angeles")

    # for transformer_cap, frequency_multiplicator, duration_multiplicator in product([10, 30, 150, 300], [0.5, 1, 10, 20], [0.25, 1, 2]):
    # charging_network = get_charging_network('simple_acn', basic_evse=True, voltage=208,
    #                                         network_kwargs={
    #                                             'station_ids': ['CA-504', 'CA-503', 'CA-502', 'CA-501'],
    #                                              #'station_ids': ['CA-501'],
    # "aggregate_cap": 32 * 208 / 1000})

    charging_network = get_charging_network('caltech', basic_evse=True, voltage=208,
                                            network_kwargs={"transformer_cap": transformer_cap})

    battery_generator = CustomizableBatteryGenerator(
        voltage=208,
        period=1,
        battery_types=[
            Linear2StageBattery],
        max_power_function="normal",
    )

    # ev_generator = RealWorldGenerator(battery_generator=battery_generator, site='caltech', period=1)
    ev_generator = get_standard_generator(
        'caltech', battery_generator, seed=42, frequency_multiplicator=frequency_multiplicator, duration_multiplicator=duration_multiplicator)

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

    df_pv = read_pv_data("../pv_150kW.csv")
    df_pv.describe()

    # In[ ]:

    # Scale the PV to fit the transformer cap
    max_P = 150 * 1000
    max_T = transformer_cap * 1000

    ratio = max_T / max_P
    df_pv.P = df_pv.P * ratio
    df_pv.describe()

    # In[ ]:

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
        # charging_reward(),
        soft_charging_reward_pv_weighted(
            df_pv, transformer_cap=transformer_cap),
    ]

    rewards_3 =[
        charging_reward(),
    ]

    rewards_2 =[
        grid_use_penalty(df_pv),
        charging_reward(),
    ]

    rewards_1 =[
        pv_utilization_reward(df_pv),
        unused_pv_penalty(df_pv),
        charging_reward(),
    ]

    rewards_0 = [
        soft_charging_reward_pv_weighted(
            df_pv, transformer_cap=transformer_cap),
    ]

    reward_setting = 0
    # TODO IF ELSE

    # In[ ]:

    # import dill as pickle

    # with open("../caltech_#stations=54_#days=7_#intervals=46_seed=8734956.pkl", "rb") as file:
    #     train_generator = pickle.load(file)

    train_generator.seed = 8734956
    _ = train_generator.reset()

    iter = 0

    while train_generator._current_date != train_generator.start_date:
        _ = train_generator.next()

        ic(iter)
        ic(train_generator._current_date)
        iter += 1

    # TODO Save the generator

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

    args = Args(
        exp_name=f"search_cap={transformer_cap}_f={frequency_multiplicator}_d={duration_multiplicator}",
        total_timesteps=steps_per_epoch * 12,
        num_steps=steps_per_epoch,
        num_envs=1,
        ent_coef=1e-4,
        # wandb:
        track=True,
        wandb_project_name="cleanRL_test",
        wandb_entity="tsturm-university-kassel",
        wandb_group="search",
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

    # In[ ]:

    from gymportal.evaluation import ACNSchedule
    from acnportal.algorithms import UncontrolledCharging, SortedSchedulingAlgo, last_come_first_served, \
        first_come_first_served

    models = {
        "FCFS": ACNSchedule(SortedSchedulingAlgo(first_come_first_served)),
        "LCFS": ACNSchedule(SortedSchedulingAlgo(last_come_first_served)),
        "Uncontrolled": ACNSchedule(UncontrolledCharging()),
    }

    models

    # In[ ]:

    from src.run_simulation import run_simulations

    df = run_simulations(models, metrics=args.eval_metrics,
                         config=args.eval_config, seed=args.eval_config["simgenerator"].seed)

    df

    # In[ ]:

    df.to_csv(f"{args.exp_name}.csv")


tyro.cli(main)
