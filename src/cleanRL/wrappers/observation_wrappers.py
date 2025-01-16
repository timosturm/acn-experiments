import gymnasium as gym
import gymnasium.spaces as spaces


class MARLObservationFlatten(gym.ObservationWrapper, gym.utils.RecordConstructorArgs):
    """
        Flattens a dict observation for MARL;
        i.e., Dict(key: nested_space, key: nested_space) -> Dict(key: flat_space, key: flat_space)
    """

    def __init__(self, env: gym.Env):
        assert isinstance(env.observation_space, spaces.Dict), \
            "This wrapper only works for MARL observations where the base space is a `spaces.Dict`!"

        gym.ObservationWrapper.__init__(self, env)

    @property
    def space_original(self) -> spaces.Dict:
        return super().observation_space

    @property
    def observation_space(self):
        return spaces.Dict(
            {key: spaces.flatten_space(space)
             for key, space in self.space_original.items()}
        )

    def observation(self, observation):
        """Flattens the observation.

        Args:
            observation: The observation to transform

        Returns:
            The transformed observation
        """
        return {
            key: spaces.flatten(self.space_original[key], obs) for key, obs in observation.items()
        }
