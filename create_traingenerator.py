#!/usr/bin/env python
# coding: utf-8

# ### How to use this:
#
# 1. Configure the charging_network, battery_generator, ev_generator and train_generator.
# 2. Run `jupyter nbconvert --to python create_traingenerator.ipynb`, to convert this notebook to a python script.
# 3. Run `nohup python create_traingenerator.py &`, to run this script in detached mode; output will be written to `nohup.out`.
# 4. The resulting sim_generator is stored as a `.pkl` file that is named after some interesting properties of it, e.g.,
#     `caltech_#stations=54_#days=7_#intervals=46_seed=8734956.pkl`.
# 5. Copy this file to the folder you want to use it in and load the generator:
#     ```python
#     import dill as pickle
#     with open(file_path, "wb") as file:
#         train_generator = pickle.load(file)
#     ```
# 6. Make sure to reset the environment once with the same seed as the train_generator:
#     `env.reset(seed=env.env.simgenerator.seed)`, otherwise the enviroment will be reset with a random seed and the generated simulations will be deleted.

# In[1]:

from tqdm import tqdm
import pytz
from datetime import datetime, timedelta
from gymportal.sim import get_charging_network, Recomputer, EvaluationSimulator, SimGenerator
from gymportal.data.battery_generators import CustomizableBatteryGenerator
from acnportal.acnsim import Linear2StageBattery
from gymportal.data.ev_generators import get_standard_generator, RealWorldGenerator
import tyro
from icecream import ic
import os
from gymportal.environment import *

# This is importent when we want to call this as a python script, because jupyter naturally has a higher recursion depth
import sys
sys.setrecursionlimit(3000)

# Print the PID when using nohup
ic(os.getpid())

# In[ ]:


timezone = pytz.timezone("America/Los_Angeles")


# for transformer_cap in tqdm([5, 10, 20, 50, 80, 100, 150, 200]):
def main(transformer_cap: int):
    # charging_network = get_charging_network('simple_acn', basic_evse=True, voltage=208,
    #                                         network_kwargs={
    #                                             'station_ids': ['CA-504', 'CA-503', 'CA-502', 'CA-501'],
    #                                              #'station_ids': ['CA-501'],
    # "aggregate_cap": 32 * 208 / 1000})

    # transformer_cap = 150
    charging_network = get_charging_network('caltech', basic_evse=True, voltage=208,
                                            network_kwargs={"transformer_cap": transformer_cap})

    battery_generator = CustomizableBatteryGenerator(
        voltage=208,
        period=1,
        battery_types=[
            Linear2StageBattery],
        max_power_function="normal",
    )

    ev_generator = RealWorldGenerator(
        battery_generator=battery_generator, site='caltech', period=1)
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

    train_generator.seed = 8734956
    ic(train_generator.start_date)
    ic(train_generator._current_date)
    _ = train_generator.reset()
    ic(train_generator._current_date)
    ic(train_generator._current_date != train_generator.start_date)

    iter = 0

    while train_generator._current_date != train_generator.start_date:
        _ = train_generator.next()

        ic(iter)
        ic(train_generator._current_date)
        iter += 1

    # In[4]:

    ic(train_generator._current_date == train_generator.start_date)

    # In[ ]:

    import dill as pickle

    file_path = f"caltech_#stations={len(charging_network.station_ids)}_#days={train_generator._simulation_days}_#intervals={train_generator._n_intervals}_transformer_cap={transformer_cap}_seed={train_generator.seed}.pkl"

    # In[6]:

    with open(file_path, "wb") as file:
        pickle.dump(train_generator, file)

    # In[ ]:

    del train_generator

    # In[7]:

    with open(file_path, "rb") as file:
        loaded = pickle.load(file)

    loaded

    # In[8]:

    train_generator = loaded

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
    ]

    reward_objects = [
        # current_constraint_violation(),
        soft_charging_reward(),
        # constraint_charging_reward(),
        # unplug_penalty(),
        # pilot_charging_rate_difference_penalty(),
    ]

    # In[ ]:

    from src.actions import ranking_schedule

    train_config = {"observation_objects": observation_objects, "action_object": ranking_schedule(),
                    "reward_objects": reward_objects,
                    "simgenerator": train_generator,
                    "meet_constraints": True}

    eval_config = train_config | {'simgenerator': eval_generator}
    validation_config = train_config | {'simgenerator': validation_generator}

    # In[ ]:

    from gymportal.environment import SingleAgentSimEnv

    env = SingleAgentSimEnv(train_config)
    env.unwrapped.simgenerator.seed

    # In[ ]:

    assert None not in env.unwrapped.simgenerator._sim_memory, "_sim_memory is not fully initialized!"

    # In[ ]:

    env.reset(seed=env.env.simgenerator.seed)
    assert None not in env.unwrapped.simgenerator._sim_memory, "_sim_memory was overwritten after resetting the environment!"


tyro.cli(main)
