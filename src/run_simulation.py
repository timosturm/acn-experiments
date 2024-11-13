from copy import deepcopy

import icecream
import pandas as pd
from gymportal import SingleAgentSimEnv
from gymportal.environment import single_charging_schedule
from gymportal.evaluation import *
import pytorch_lightning
from tqdm import tqdm

from src.utils import CustomSchedule, FlattenSimEnv

metrics = {
    "SoC >= 90%": percentage_soc,
    "mean SoC": mean_soc,
    "median SoC": median_soc,
    "prop feasible steps": proportion_of_feasible_charging,
    "prop feasible charge": proportion_of_feasible_charge
}


def run_simulations(models: Dict[str, CanSchedule], metrics: Dict[str, Callable], config,
                    change_action_object_automatically: bool = True, seed: Optional[int] = None) -> pd.DataFrame:
    configs = {
        key: deepcopy(config) | {"action_object": single_charging_schedule()} if isinstance(value,
                                                                                            ACNSchedule) else deepcopy(
            config)
        for key, value in models.items()
    }

    sims = {}
    for algo_name, scheduler in tqdm(models.items(), desc="Models"):
        if isinstance(scheduler, CustomSchedule):
            env_type = FlattenSimEnv
        else:
            env_type = SingleAgentSimEnv

        sims[algo_name] = evaluate_model(scheduler, env_type=env_type, env_config=configs[algo_name],
                                         seed=seed)

    results = {metric_name: [m(s) for s in sims.values()]
               for metric_name, m in metrics.items()}
    
    results_df = pd.DataFrame.from_dict(results)
    results_df["Algorithms"] = sims.keys()
    results_df.set_index("Algorithms", inplace=True)
    return results_df
