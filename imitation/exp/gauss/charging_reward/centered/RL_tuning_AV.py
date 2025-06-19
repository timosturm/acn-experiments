import json
from optuna.pruners import MedianPruner
import torch
from tqdm import tqdm
from src.actions import beta_schedule_normalized, beta_ranking_plus
from src.cleanRL.agent import Agent, BetaAgent, BetaNormAgent
from src.cleanRL.environment import make_env
from src.data import get_data, get_gmm, get_pv_data
from src.observations import minute_observation_stay
from src.pv.metrics import *
from gymportal.evaluation import *
from src.pv.rewards import *
from src.pv.observations import pv_observation_mean, pv_observation_mean_normalized, pv_observation_normalized
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
from src.imitation.objective import objective_IL, objective_RL, objective_combined
from src.rewards import sparse_soc_reward
from src.utils import AV_pod_ids, get_generator, get_power_function, get_steps_per_epoch


# This is importent when we want to call this as a python script, because jupyter naturally has a higher recursion depth
sys.setrecursionlimit(3000)

# Print the PID when using nohup
ic(os.getpid())


timezone = pytz.timezone("America/Los_Angeles")

charging_network = get_charging_network(
    'simple_acn',
    basic_evse=True,
    voltage=208,
    network_kwargs={
        'station_ids': AV_pod_ids,
        "aggregate_cap": (150 / 54) * len(AV_pod_ids),
    },
)

# charging_network = get_charging_network('caltech', basic_evse=True, voltage=208,
#                                         network_kwargs={"transformer_cap": 150})

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
    site='caltech',
    model=get_gmm(),
    battery_generator=battery_generator,
    seed=42,
    frequency_multiplicator=10,
    duration_multiplicator=1,
    data=get_data(),
)

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


df_pv = get_pv_data()
df_pv.describe()
df_pv.P /= 54 / len(charging_network.station_ids)

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
    pv_observation_mean_normalized(df_pv),
    pv_observation_normalized(df_pv),
]

reward_objects = [
    pv_utilization_reward(df_pv),
    unused_pv_reward(df_pv),
    # sparse_soc_reward(),
    charging_reward(),
]

train_generator.seed = 8734956
_ = train_generator.reset()

steps_per_epoch = ic(
    get_steps_per_epoch(train_generator)
)

train_config = {
    "observation_objects": observation_objects,
    "action_object": beta_schedule_normalized(),
    "reward_objects": reward_objects,
    "simgenerator": train_generator,
    "meet_constraints": True,
}

validation_config = train_config | {'simgenerator': validation_generator}
test_config = train_config | {'simgenerator': test_generator}

metrics = {
    "SoC >= 90%": percentage_soc,
    "mean SoC": mean_soc,
    "median SoC": median_soc,
    # "prop feasible steps": proportion_of_feasible_charging,
    # "prop feasible charge": proportion_of_feasible_charge,
    # "pv utilization": lambda sim: pv_utilization_metric(sim, df_pv),
    "grid usage": lambda sim: grid_use_metric(sim, df_pv),
    "unused PV": lambda sim: unused_pv_metric(sim, df_pv),
}

# prepare loading the best model
# name = "Imitation_best"
# best_state_dict = torch.load(f"{name}.mdl", weights_only=True)
# with open(f"{name}.json", 'rb') as file:
#     js = json.loads(file.read())
#     hiddens = [v for k, v in js["parameter"].items() if "_layer_" in k]

study_name: str = "gauss_charging_centered_AV"
hiddens = [2048, 512, 128]

args = MyArgs(
    exp_name="RL",
    wandb_project_name="rl",
    wandb_group=study_name,
    seed=42,  # TODO
    imitation=ImitationArgs(
        # TODO Store baseline as a parameter
        train_ds="AV_46_weeks_training.parquet.gzip",
        validation_ds="AV_46_weeks_validation.parquet.gzip",
        agent_class=Agent,
    ),
    eval=EvalArgs(
        make_env=lambda: make_env(validation_config, 0.99, 0, 930932)(),
        metrics=metrics,
        hiddens=hiddens,
        agent_class=Agent,
    ),
    rl=RLArgs(
        total_timesteps=steps_per_epoch * 16,
        config=train_config,
        metrics=metrics,
        # state_dict=best_state_dict,
        hiddens=hiddens,
        agent_class=Agent,
    ),
)

if __name__ == "__main__":
    study = optuna.create_study(
        study_name=study_name,
        storage=f'sqlite:///{study_name}.db',
        load_if_exists=True,
        direction="maximize",
        # sampler=
        pruner=MedianPruner(
            n_startup_trials=10,
            n_warmup_steps=5,
        )
    )

    def objective_wrapper(trial):
        return objective_RL(
            trial,
            args,
            make_env,
        )

    study.optimize(objective_wrapper, n_trials=1)  # TODO Add more trials

    # TODO Do something with the best model here (?)
