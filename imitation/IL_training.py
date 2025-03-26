from optuna.pruners import MedianPruner
from src.cleanRL.environment import make_env
from src.observations import minute_observation_stay
from src.pv.metrics import *
from gymportal.evaluation import *
from src.pv.rewards import *
from src.pv.observations import pv_observation_mean
from gymportal.environment import *
from src.pv.pv import read_pv_data
import pytz
from datetime import datetime, timedelta
from gymportal.sim import get_charging_network, Recomputer, EvaluationSimulator, SimGenerator
from gymportal.data.battery_generators import CustomizableBatteryGenerator
from acnportal.acnsim import Linear2StageBattery
from gymportal.data.ev_generators import get_standard_generator, RealWorldGenerator
import sys
import os
import optuna
from icecream import ic
from src.imitation.args import MyArgs, ImitationArgs, RLArgs, EvalArgs
from objective import objective_IL, objective_combined
from src.utils import get_generator


# This is importent when we want to call this as a python script, because jupyter naturally has a higher recursion depth
sys.setrecursionlimit(3000)

# Print the PID when using nohup
ic(os.getpid())


timezone = pytz.timezone("America/Los_Angeles")


# charging_network = get_charging_network('simple_acn', basic_evse=True, voltage=208,
#                                         network_kwargs={
#                                             'station_ids': ['CA-504', 'CA-503', 'CA-502', 'CA-501'],
#                                             # 'station_ids': ['CA-501'],
#                                             "aggregate_cap": 32 * 208 / 1000})

charging_network = get_charging_network('caltech', basic_evse=True, voltage=208,
                                        network_kwargs={"transformer_cap": 150})

battery_generator = CustomizableBatteryGenerator(
    voltage=208,
    period=1,
    battery_types=[
        Linear2StageBattery],
    max_power_function="normal",
)

# ev_generator = RealWorldGenerator(battery_generator=battery_generator, site='caltech', period=1)
ev_generator = get_generator(
    'caltech',
    "../triple_gmm+sc.pkl",
    battery_generator,
    seed=42,
    frequency_multiplicator=10,
    duration_multiplicator=2,
    file_path="../caltech_2018-03-25 00:00:00-07:53_2020-05-31 00:00:00-07:53_False.csv"
)

# TODO Use time intervals and GMMs from https://github.com/chrisyeh96/sustaingym/blob/main/sustaingym/envs/evcharging/utils.py
# I.e., train on generated data, evaluate on new generated data and real data from the same interval
# optional: compare to "out-of-distribution" data from different interval

train_generator = SimGenerator(
    charging_network=charging_network,
    simulation_days=1,
    n_intervals=7 * 46,
    start_date=timezone.localize(datetime(2019, 1, 1)),
    ev_generator=ev_generator,
    recomputer=Recomputer(recompute_interval=10, sparse=True),
    sim_class=EvaluationSimulator,
)

ic(train_generator.end_date + timedelta(days=1))

validation_generator = SimGenerator(
    charging_network=charging_network,
    simulation_days=7,
    n_intervals=1,
    start_date=train_generator.end_date + timedelta(days=1),
    ev_generator=ev_generator,
    recomputer=Recomputer(recompute_interval=10, sparse=True),
    sim_class=EvaluationSimulator,
)

ic(validation_generator.end_date + timedelta(days=1))

test_generator = SimGenerator(
    charging_network=charging_network,
    simulation_days=14,
    n_intervals=1,
    start_date=validation_generator.end_date + timedelta(days=1),
    ev_generator=ev_generator,
    recomputer=Recomputer(recompute_interval=10, sparse=True),
    sim_class=EvaluationSimulator,
)

ic(test_generator.end_date + timedelta(days=1))


df_pv = read_pv_data("../pv_150kW.csv")
df_pv.describe()


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
    # grid_use_penalty(df_pv),
    unused_pv_penalty(df_pv),
    charging_reward(),
    # soft_charging_reward_pv_weighted(df_pv, transformer_cap=150),
]

train_generator.seed = 8734956
_ = train_generator.reset()

iter = 0

while train_generator._current_date != train_generator.start_date:
    _ = train_generator.next()

    ic(iter)
    ic(train_generator._current_date)
    iter += 1

steps_per_epoch = 0
for eval_sim in train_generator._sim_memory:
    steps_per_epoch += len(eval_sim.event_queue.queue)

ic(steps_per_epoch)

train_config = {"observation_objects": observation_objects, "action_object": zero_centered_single_charging_schedule_normalized(),
                "reward_objects": reward_objects,
                "simgenerator": train_generator,
                "meet_constraints": True}

validation_config = train_config | {'simgenerator': validation_generator}
test_config = train_config | {'simgenerator': test_generator}

metrics = {
    "SoC >= 90%": percentage_soc,
    "mean SoC": mean_soc,
    "median SoC": median_soc,
    # "prop feasible steps": proportion_of_feasible_charging,
    # "prop feasible charge": proportion_of_feasible_charge,
    "pv utilization": lambda sim: pv_utilization_metric(sim, df_pv),
    "grid usage": lambda sim: grid_use_metric(sim, df_pv),
    "unused pv": lambda sim: unused_pv_metric(sim, df_pv),
}

args = MyArgs(
    exp_name="Imitation",
    wandb_project_name="imitation",
    wandb_group="test run",
    seed=42,  # TODO
    imitation=ImitationArgs(
        # TODO Store baseline as a parameter
        train_ds="FCFS_gen_triple_46_weeks_training.parquet.gzip",
        validation_ds="FCFS_gen_triple_46_weeks_validation.parquet.gzip",
    ),
    eval=EvalArgs(
        make_env=lambda: make_env(validation_config, 0.99, 0, 930932)(),
        metrics=metrics,
    ),
    rl=RLArgs(
        total_timesteps=steps_per_epoch * 20,
        config=train_config,
        metrics=metrics,
    ),
)

if __name__ == "__main__":
    study = optuna.create_study(
        study_name='IL-tuning',
        storage='sqlite:///IL-tuning.db',
        load_if_exists=True,
        direction="maximize",
        # sampler=
        pruner=MedianPruner(
            n_startup_trials=10,
            n_warmup_steps=5,
        )
    )

    def objective_wrapper(trial):
        return objective_IL(
            trial,
            args,
            make_env,
        )

    study.optimize(objective_wrapper, n_trials=1)  # TODO Add more trials

    # TODO Do something with the best model here (?)
