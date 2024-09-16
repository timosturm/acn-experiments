import os

from ray import tune
from ray.air import RunConfig, CheckpointConfig, FailureConfig
from ray.rllib.algorithms import Algorithm
from ray.tune import TuneConfig
from ray.tune.stopper import CombinedStopper, MaximumIterationStopper
from ray.util.client import ray

from data.ev_generators import get_persistent_folder
from experiments.ppo_config import ppo_config

os.environ["TUNE_MAX_PENDING_TRIALS_PG"] = "60"

ray.init(_memory=70 * 1024 ** 3, num_cpus=60, num_gpus=0, include_dashboard=False, ignore_reinit_error=True)

config = ppo_config

parameters = {
    "lr": tune.grid_search([1e-3, 1e-4, 1e-5, 1e-6, 1e-7]),
    "gamma": tune.grid_search([0.8, 0.85, 0.9, 0.95, 0.99]),
    # "lambda": tune.grid_search([0.9, 0.95, 1]),
    # "clip_param": tune.grid_search([0.1, 0.2, 0.3, 0.4, 0.5])
}

stopper = CombinedStopper(
    MaximumIterationStopper(max_iter=100),
)

metric = "episode_reward_mean"

tuner = tune.Tuner(
    config.build().__class__,
    # storage_path='./tune_results',
    run_config=RunConfig(stop=stopper,
                         checkpoint_config=CheckpointConfig(
                             checkpoint_score_attribute=metric,
                             checkpoint_frequency=1,
                             num_to_keep=1,
                             checkpoint_at_end=True),
                         failure_config=FailureConfig(max_failures=3),
                         storage_path=get_persistent_folder("ray_results/single", __file__),
                         ),
    tune_config=TuneConfig(
        # scheduler=scheduler,
        num_samples=1,
        reuse_actors=True
    ),
    param_space=config.__dict__ | parameters
)

_ = tuner.fit()

results = tuner.get_results()
best_result = results.get_best_result(metric="episode_reward_mean", mode="max")
checkpoint = best_result.checkpoint
algo = Algorithm.from_checkpoint(checkpoint)

algo.save(f"{algo}_grid_check")
