from gymportal.plotting.plotting import plot_sim_evaluation
from gymportal.evaluation import evaluate_model
from datetime import datetime
import wandb
from src.utils import CustomSchedule
from gymportal.environment import *
import pandas as pd
from pytorch_lightning.callbacks import Callback
from gymportal.sim import Recomputer, EvaluationSimulator, SimGenerator
from datetime import datetime
from src.run_simulation import run_simulations
from src.utils import FlattenSimEnv


class EvaluationMetricsCallback(Callback):
    def __init__(self, models, metrics, config, seed, run):
        super().__init__()
        self.models = models
        self.metrics = metrics
        self.config = config
        self.seed = seed
        self.run = run

        self.state = {"df_before": None, "df_after": None}

    @property
    def df_diff(self):
        df_before, df_after = self.state["df_before"], self.state["df_after"]

        return pd.DataFrame(columns=df_before.columns,
                            data=df_after.to_numpy() - df_before.to_numpy(),
                            index=df_before.index)

    def on_train_epoch_start(self, trainer, pl_module):
        df = run_simulations(self.models | {"PPO": CustomSchedule(pl_module)}, metrics=self.metrics,
                             config=self.config, seed=self.seed)
        self.state["df_before"] = df

        self.run.log({"df_before": wandb.Table(dataframe=df)})

    def on_train_epoch_end(self, trainer, pl_module):
        df = run_simulations(self.models | {"PPO": CustomSchedule(pl_module)}, metrics=self.metrics,
                             config=self.config, seed=self.seed)
        self.state["df_after"] = df

        self.run.log({"df_after": wandb.Table(dataframe=df)})
        self.run.log({"df_diff": wandb.Table(dataframe=self.df_diff)})


class EvaluationFigureCallback(Callback):
    def __init__(self, charging_network, timezone, ev_generator, train_config, run):
        self.run = run

        eval_generator = SimGenerator(
            charging_network=charging_network,
            simulation_days=1,
            n_intervals=1,
            start_date=timezone.localize(datetime(2019, 9, 23)),
            ev_generator=ev_generator,
            recomputer=Recomputer(recompute_interval=10, sparse=True),
            sim_class=EvaluationSimulator,
        )

        self.state = {"config": train_config |
                      {"simgenerator": eval_generator}}

    def on_train_end(self, trainer, pl_module):
        eval_sim = evaluate_model(CustomSchedule(
            pl_module), env_type=FlattenSimEnv, env_config=self.state["config"])

        fig = plot_sim_evaluation(eval_sim)  # .savefig("evaluation.png")

        fig.canvas.draw()

        # Now we can save it to a numpy array.
        data = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
        data = data.reshape(fig.canvas.get_width_height()[::-1] + (4,))

        self.run.log({"evaluation": wandb.Image(data)})
