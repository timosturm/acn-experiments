#!/usr/bin/env python
# coding: utf-8

# In[1]:

if __name__ == "__main__":

    from datetime import datetime, timedelta
    from tqdm import tqdm
    from ray.util.client import ray

    from run_simulation import run_simulations, metrics

    ray.init(_memory=16 * 1024 ** 3, num_cpus=8, num_gpus=0, include_dashboard=True, ignore_reinit_error=True)

    # In[2]:

    from gymportal.data.ev_generators import get_standard_generator, RealWorldGenerator
    from acnportal.acnsim import Linear2StageBattery
    from gymportal.data.battery_generators import CustomizableBatteryGenerator
    from gymportal.sim import get_charging_network, Recomputer, EvaluationSimulator, SimGenerator
    from icecream import ic

    # charging_network = get_charging_network('simple_acn', basic_evse=True, voltage=208,
    #                                         network_kwargs={
    #                                             'station_ids': ['CA-504', 'CA-503', 'CA-502', 'CA-501'],
    #                                             'station_ids': ['CA-501'],
    # "aggregate_cap": 32 * 208 / 1000})

    charging_network = get_charging_network('caltech', basic_evse=True, voltage=208,
                                            network_kwargs={"transformer_cap": 40 * 32 * 208 / 1000})

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
        start_date=datetime(2019, 1, 1),
        ev_generator=ev_generator,
        recomputer=Recomputer(recompute_interval=10, sparse=True),
        sim_class=EvaluationSimulator,
    )

    ic(train_generator.end_date + timedelta(days=1))

    eval_generator = SimGenerator(
        charging_network=charging_network,
        simulation_days=14,
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
        # pilot_charging_rate_difference_penalty(),
    ]

    # In[4]:

    from extra import unplug_penalty, ranking_schedule

    train_config = {"observation_objects": observation_objects,
                    "action_object": zero_centered_single_charging_schedule_normalized(),
                    "reward_objects": [unplug_penalty()],
                    "simgenerator": train_generator,
                    "meet_constraints": True}

    eval_config = train_config | {'simgenerator': eval_generator}
    validation_config = train_config | {'simgenerator': validation_generator}

    # In[5]:

    from gymportal.auxilliaries.callbacks import MetricsCallback
    from ray.rllib.algorithms.ppo import PPOConfig

    # lr_schedule = [[0, 5e-5], [int(1e20), 5e-ranking]]

    config = (
        PPOConfig()
        .environment('single_agent_env', env_config=train_config)
        .rollouts(num_rollout_workers=1)
        .framework("tf2")
        .evaluation(evaluation_num_workers=1, evaluation_interval=3)
        .training(use_kl_loss=False, kl_coeff=0.0)
        # .training(lr_schedule=lr_schedule)
        # .exploration(explore=False, exploration_config={"type": "StochasticSampling"})
        .callbacks(MetricsCallback)
    )

    # In[6]:

    algo = config.build()

    # In[7]:

    from gymportal.evaluation import ACNSchedule, RllibSchedule
    from acnportal.algorithms import UncontrolledCharging, SortedSchedulingAlgo, last_come_first_served, \
        first_come_first_served

    models = {
        "PPO": RllibSchedule(algo),
        "FCFS": ACNSchedule(SortedSchedulingAlgo(first_come_first_served)),
        "LCFS": ACNSchedule(SortedSchedulingAlgo(last_come_first_served)),
        "Uncontrolled": ACNSchedule(UncontrolledCharging()),
    }

    # In[8]:

    df = run_simulations(models, metrics=metrics, config=validation_config)
    df

    # In[ ]:

    df.to_csv("before.csv")
    ax = df.plot.bar()
    fig = ax.get_figure()
    fig.savefig("before.png", dpi=600)

    # In[ ]:

    res = []

    for i in tqdm(range(12), desc='training iteration'):
        res.append(algo.train())

        algo.save(checkpoint_dir=f"{algo}_check")

    # In[ ]:

    from gymportal.plotting.plotting import plot_training

    _ = plot_training(res).savefig(f"{algo}", dpi=600)

    # In[ ]:

    # plain     31:28
    # cython    33:21
    # v2        31:58

    # In[ ]:

    # from ray.rllib.algorithms import Algorithm
    # #algo = Algorithm.from_checkpoint(checkpoint_dir + "/checkpoint_000001")
    #
    # algo = Algorithm.from_checkpoint("PPO_check")

    # In[ ]:

    # algo.evaluate()

    # In[ ]:

    eval_generator = SimGenerator(
        charging_network=charging_network,
        simulation_days=1,
        n_intervals=1,
        start_date=datetime(2019, 9, 23),
        ev_generator=ev_generator,
        recomputer=Recomputer(recompute_interval=10, sparse=True),
        sim_class=EvaluationSimulator,
    )

    evaluation_config = train_config | {"simgenerator": eval_generator}

    # In[ ]:

    from gymportal.evaluation import evaluate_model

    eval_sim = evaluate_model(RllibSchedule(algo), env_type=SingleAgentSimEnv, env_config=evaluation_config)

    # In[ ]:

    from gymportal.plotting.plotting import plot_sim_evaluation

    _ = plot_sim_evaluation(eval_sim)

    # In[ ]:

    df = run_simulations(models, metrics=metrics, config=validation_config)
    df

    # In[ ]:

    df.to_csv("after.csv")
    ax = df.plot.bar()
    fig = ax.get_figure()
    fig.savefig("after.png", dpi=600)
