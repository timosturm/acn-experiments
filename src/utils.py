from datetime import datetime
from sklearn.mixture import BayesianGaussianMixture
import pytz
from gymportal.auxilliaries.file_utils import get_persistent_folder
from gymportal.data.ev_generators import SklearnGenerator, get_data, extract_training_data
import pickle
import numpy as np
from sklearn.base import TransformerMixin, BaseEstimator
import gymnasium as gym
from acnportal.acnsim import Simulator
from itertools import tee
from typing import Optional
from gymportal.evaluation import CanSchedule

import gymnasium.spaces as spaces
from gymnasium.wrappers import FlattenObservation
from gymportal.environment import SingleAgentSimEnv
from gymportal.auxilliaries.interfaces_custom import EvaluationGymTrainingInterface


class FlattenSimEnv(FlattenObservation):

    env: SingleAgentSimEnv

    def __init__(self, config, iface_type=EvaluationGymTrainingInterface):
        self.env = SingleAgentSimEnv(config, iface_type)
        self.observation_space = spaces.flatten_space(
            self.env.observation_space)

    def observation(self, observation):
        """Flattens an observation.

        Args:
            observation: The observation to flatten

        Returns:
            The flattened observation
        """
        return spaces.flatten(self.env.observation_space, observation)


def _pairwise(iterable):
    """
    Taken from https://docs.python.org/3/library/itertools.html#itertools.pairwise, because the servers python
    version does not yet have this from itertools.

    Args:
        iterable:

    Returns:

    """
    # pairwise('ABCDEFG') --> AB BC CD DE EF FG
    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)


def evaluate_model(model: CanSchedule, eval_env: gym.Env, seed: Optional[int] = None) -> Simulator:
    """
    Evaluates a model / algorithm (either from stable_baselines3 or acnportal) by running a simulation.
    In the case of stable_baselines3 models, the predictions are made deterministically.

    Args:
        seed:
            Optional seed to make evaluations reproducible.
        env_type:
            The type of environment to use, either a single- or multi-agent environment.
        model:
            The model to produce pilot signals.
        env_config:
            Configuration dict containing rewards, actions, observations, and an interface_generating_function.
            See RebuildingEnvV2Config for details.

    Returns:
        Simulation after completion.
    """
    done = False
    observation, _ = eval_env.reset(seed=seed)
    agg_reward = 0

    while not done:

        iface = eval_env.unwrapped.interface
        action = model.get_action(observation, iface)

        observation, rew, terminated, truncated, _ = eval_env.step(
            action)

        agg_reward += rew

        # if isinstance(eval_env, MultiAgentEnv):
        #     done = terminated['__all__'] or truncated['__all__']
        # else:
        done = terminated or truncated

    # Get the simulator we want to return
    evaluation_simulation = eval_env.unwrapped.interface._simulator

    return evaluation_simulation, agg_reward


class ManualMaxScaler(BaseEstimator, TransformerMixin):
    def __init__(self, max_values):
        self.max_values = np.array(max_values)

    def fit(self, X, y=None):
        X = np.asarray(X)
        if X.shape[1] != len(self.max_values):
            raise ValueError(
                f"Expected {X.shape[1]} max values, but got {len(self.max_values)}")
        return self  # No fitting needed

    def transform(self, X):
        X = np.asarray(X)
        return X / self.max_values  # Element-wise division

    def inverse_transform(self, X_scaled):
        return X_scaled * self.max_values  # Multiply back by max values


class ScalableSklearnGenerator(SklearnGenerator):

    def __init__(
        self,
        period,
        battery_generator,
        model,
        scaler,
        frequencies_per_hour,
        duration_multiplicator=1,
        arrival_min=0,
        arrival_max=24,
        duration_min=0.0833,
        duration_max=48,
        energy_min=0.5,
        energy_max=150,
        seed=None
    ):
        super().__init__(
            period,
            battery_generator,
            model,
            frequencies_per_hour,
            duration_multiplicator,
            arrival_min, arrival_max,
            duration_min, duration_max,
            energy_min,
            energy_max,
            seed
        )

        self.scaler = scaler

    def _sample(self, n_samples: int):
        """ Generate random samples from the fitted model.

        Args:
            n_samples (int): Number of samples to generate.

        Returns:
            np.ndarray: shape (n_samples, 3), randomly generated samples. Column 1 is
                the arrival time in hours since midnight, column 2 is the session duration in hours,
                and column 3 is the energy demand in kWh.
        """
        if n_samples > 0:
            ev_matrix, _ = self.sklearn_model.sample(n_samples)
            ev_matrix = self.scaler.inverse_transform(ev_matrix)
            return self._clip_samples(ev_matrix)
        else:
            return np.array([])


def get_generator(site, model_path: str, battery_generator, token: Optional[str] = None, seed: Optional[int] = None,
                  frequency_multiplicator=10, duration_multiplicator=1):
    """

    Args:
        site: The site which is used as a data source for the generative model.
        battery_generator: The generator for EV batteries.
        token: The token to access acn-data.
        seed: A seed for random number generator
        frequency_multiplicator: A multiplicator for the arrival frequencies of EVs, e.g., a higher value makes it
            more likely for an EV to arrive at a given point in time.

    Returns:

    """
    timezone = pytz.timezone('America/Los_Angeles')
    data = get_data(
        site,
        token,
        drop_columns=(),
        start=datetime(2018, 3, 25, tzinfo=timezone),
        end=datetime(2020, 5, 31, tzinfo=timezone)
    )
    X = extract_training_data(data)

    try:
        with open(model_path, "rb") as f:
            gmm, scaler = pickle.load(f)
    except FileNotFoundError:
        print(f"No existing GMM found for site={site}!")

    connection_time = X[:, 0]

    frequencies, _ = np.histogram(connection_time, bins=range(0, 25, 1))
    frequencies = np.array(frequencies) / np.sum(frequencies)

    generator = ScalableSklearnGenerator(
        period=1,
        model=gmm,
        scaler=scaler,
        frequencies_per_hour=frequencies * frequency_multiplicator,
        battery_generator=battery_generator,
        duration_multiplicator=duration_multiplicator,
        seed=seed
    )

    return generator
