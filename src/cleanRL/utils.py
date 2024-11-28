import gymnasium as gym
from gymportal.sim import SimGenerator, Recomputer, EvaluationSimulator
from typing import Callable, Dict, Optional
from src.cleanRL.agent import Agent
import torch
from gymportal.plotting.plotting import plot_sim_evaluation
from .scheduler import CleanRLSchedule
import wandb
import numpy as np

# from src.run_simulation import run_simulations
from copy import deepcopy

import pandas as pd
from gymportal.environment import single_charging_schedule
from tqdm import tqdm
from src.utils import evaluate_model
from src.cleanRL import Args
from src.cleanRL.environment import make_env
from src.cleanRL.scheduler import CleanRLSchedule
from gymportal.evaluation import CanSchedule, ACNSchedule


def run_simulations(models: Dict[str, CanSchedule], metrics: Dict[str, Callable], config,
                    change_action_object_automatically: bool = True, seed: Optional[int] = None, ) -> pd.DataFrame:
    configs = {
        key: deepcopy(config) | {"action_object": single_charging_schedule()} if isinstance(value,
                                                                                            ACNSchedule) else deepcopy(
            config)
        for key, value in models.items()
    }

    sims = {}
    for algo_name, scheduler in tqdm(models.items(), desc="Models"):
        # if isinstance(scheduler, CustomScheduler):
        #     env_type = FlattenSimEnv
        # else:
        #     env_type = SingleAgentSimEnv

        env = make_env(configs[algo_name], seed=seed, gamma=0)()

        sims[algo_name] = evaluate_model(scheduler, eval_env=env, seed=seed)

    results = {metric_name: [m(s) for s in sims.values()]
               for metric_name, m in metrics.items()}

    results_df = pd.DataFrame.from_dict(results)
    results_df["Algorithms"] = sims.keys()
    results_df.set_index("Algorithms", inplace=True)
    return results_df


def log_metrics_table(models, args, run):

    df = run_simulations(models, metrics=args.eval_metrics,
                         config=args.eval_config, seed=args.eval_seed)

    run.log({"eval/metrics": wandb.Table(dataframe=df.reset_index())})


def log_evaluation_plot(agent, args, run):

    reference_generator = deepcopy(args.eval_config["simgenerator"])

    eval_generator = SimGenerator(
        charging_network=reference_generator._charging_network,
        simulation_days=1,
        n_intervals=1,
        start_date=reference_generator.start_date,
        ev_generator=reference_generator._ev_generator,
        recomputer=Recomputer(recompute_interval=10, sparse=True),
        sim_class=EvaluationSimulator,
    )

    cfg = deepcopy(args.eval_config) | {"simgenerator": eval_generator}

    seed = args.eval_seed
    eval_env = make_env(cfg, seed=seed, gamma=0)()

    ppo_scheduler = CleanRLSchedule(agent)
    eval_sim = evaluate_model(ppo_scheduler, eval_env=eval_env, seed=seed)

    fig = plot_sim_evaluation(eval_sim)
    # this will draw the canvas if not in a jupyter notebook
    fig.canvas.draw()

    # now we can save it to a numpy array
    data = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (4,))

    run.log({"plots/actions": wandb.Image(data)})


def load_agent(args: Args, path):
    agent = Agent(envs=gym.vector.SyncVectorEnv(
        [
            make_env(args.eval_config, args.gamma)
        ]
    ))

    agent.load_state_dict(torch.load(path, weights_only=True))

    return agent


def log_model(run, path):
    artifact = wandb.Artifact('model', type='model')
    artifact.add_file(path)
    run.log_artifact(artifact)


def get_model_path(args: Args):
    run_name = f"{args.exp_name}__{args.seed}"
    return f"runs/{run_name}/{args.exp_name}.cleanrl_model"
