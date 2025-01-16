from typing import Callable
import numpy as np
import gymnasium as gym
from gymnasium.wrappers.normalize import RunningMeanStd
from icecream import ic


class MARLTransformReward(gym.RewardWrapper, gym.utils.RecordConstructorArgs):
    """Transform the reward via an arbitrary function.

    Warning:
        If the base environment specifies a reward range which is not invariant under :attr:`f`, the :attr:`reward_range` of the wrapped environment will be incorrect.

    Example:
        >>> import gymnasium as gym
        >>> from gymnasium.wrappers import TransformReward
        >>> env = gym.make("CartPole-v1")
        >>> env = TransformReward(env, lambda r: 0.01*r)
        >>> _ = env.reset()
        >>> observation, reward, terminated, truncated, info = env.step(env.action_space.sample())
        >>> reward
        0.01
    """

    def __init__(self, env: gym.Env, f: Callable[[float], float]):
        """Initialize the :class:`TransformReward` wrapper with an environment and reward transform function :attr:`f`.

        Args:
            env: The environment to apply the wrapper
            f: A function that transforms the reward
        """
        gym.utils.RecordConstructorArgs.__init__(self, f=f)
        gym.RewardWrapper.__init__(self, env)

        assert callable(f)
        self.f = f

    def reward(self, reward):
        """Transforms the reward using callable :attr:`f`.

        Args:
            reward: The reward to transform

        Returns:
            The transformed reward
        """
        return {
            agent_id: self.f(r) for agent_id, r in reward.items()
        }


class MARLNormalizeReward(gym.core.Wrapper, gym.utils.RecordConstructorArgs):
    r"""This wrapper will normalize immediate rewards s.t. their exponential moving average has a fixed variance.

    The exponential moving average will have variance :math:`(1 - \gamma)^2`.

    Note:
        The scaling depends on past trajectories and rewards will not be scaled correctly if the wrapper was newly
        instantiated or the policy was changed recently.
    """

    def __init__(
        self,
        env: gym.Env,
        gamma: float = 0.99,
        epsilon: float = 1e-8,
    ):
        """This wrapper will normalize immediate rewards s.t. their exponential moving average has a fixed variance.

        Args:
            env (env): The environment to apply the wrapper
            epsilon (float): A stability parameter
            gamma (float): The discount factor that is used in the exponential moving average.
        """
        gym.utils.RecordConstructorArgs.__init__(
            self, gamma=gamma, epsilon=epsilon)
        gym.Wrapper.__init__(self, env)

        self.num_envs = getattr(env, "num_envs", 1)
        self.is_vector_env = getattr(env, "is_vector_env", False)
        self.return_rms = {agent_id: RunningMeanStd(
            shape=()) for agent_id in self.get_agent_ids()}
        self.returns = {agent_id: np.zeros(
            self.num_envs) for agent_id in self.get_agent_ids()}
        self.gamma = gamma
        self.epsilon = epsilon

    def step(self, action):
        """Steps through the environment, normalizing the rewards returned."""
        obs, rews, terminateds, truncateds, infos = self.env.step(action)
        # if not self.is_vector_env:
        # rews = np.array([rews])
        
        ic(rews)
        assert isinstance(rews, dict), f"expected dict, got {type(rews)}"

        ic(self.returns)
        ic(terminateds)
        
        self.returns = {
            self.returns[agent_id] * self.gamma *
            (1 - terminateds[agent_id]) + rews[agent_id]
            for agent_id in self.get_agent_ids()
        }

        rews = self.normalize(rews)

        ic(rews)

        # if not self.is_vector_env:
        #     rews = rews[0]

        return obs, rews, terminateds, truncateds, infos

    def normalize(self, rews):
        """Normalizes the rewards with the running mean rewards and their variance."""
        for agent_id in self.get_agent_ids():
            self.return_rms[agent_id].update(self.returns[agent_id])

        return {agent_id: rews[agent_id] / np.sqrt(self.return_rms[agent_id].var + self.epsilon) for agent_id in self.get_agent_ids()}
