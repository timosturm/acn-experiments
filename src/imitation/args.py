from platform import architecture
from gymnasium.vector.vector_env import VectorEnv
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional
import gymnasium as gym


@dataclass
class EvalArgs:
    make_env: Callable[[], gym.Env]
    metrics: Dict
    hiddens: List[int] = field(default_factory=lambda: [128, 128, 128])

    _env: Optional[gym.Env] = None

    @property
    def env(self) -> gym.Env:
        if not self._env:
            self._env = self.make_env()

        return self._env


@dataclass
class ImitationArgs:
    train_ds: str
    validation_ds: str

    lr: float = 1e-4
    batch_size: int = 512
    n_epochs: int = 10
    hiddens: List[int] = field(default_factory=lambda: [128, 128, 128])

    @property
    def n_hiddens(self) -> int:
        return len(self.hiddens)


@dataclass
class RLArgs:
    """The configuration of the env"""
    config: Dict = None
    metrics: Dict = None

    """The training environemnt"""
    envs: VectorEnv = None

    """the number of parallel game environments"""
    num_envs: int = 1  # Only supported value at the moment
    """the number of steps to run in each environment per policy rollout"""
    num_steps: int = 2048
    """the number of mini-batches"""
    num_minibatches: int = 32
    """total timesteps of the experiments"""
    total_timesteps: int = 1000000

    # to be filled by trial.suggest
    """the learning rate of the optimizer"""
    lr: float = 3e-4
    """coefficient of the entropy"""
    ent_coef: float = 0
    """the discount factor gamma"""
    gamma: float = 0.99
    """the lambda for the general advantage estimation"""
    gae_lambda: float = 0.95
    """the maximum norm for the gradient clipping"""
    max_grad_norm: float = 0.5
    """the maximum norm for the gradient clipping"""
    vf_coef: float = 0.5
    """the surrogate clipping coefficient"""
    clip_coef = 0.2

    """the K epochs to update the policy"""
    update_epochs: int = 10
    """Toggles advantages normalization"""
    norm_adv: bool = True
    """Toggles whether or not to use a clipped loss for the value function, as per the paper."""
    clip_vloss: bool = True
    """Toggle learning rate annealing for policy and value networks"""
    anneal_lr: bool = True

    # to be filled in runtime
    """the batch size (computed in runtime)"""
    batch_size: int = 0
    """the mini-batch size (computed in runtime)"""
    minibatch_size: int = 0
    """the number of iterations (computed in runtime)"""
    num_iterations: int = 0
    """number of neurons per hidden layer"""
    hiddens: List[int] = field(default_factory=lambda: [128, 128, 128])

    target_kl = None

    state_dict: Optional[dict] = None


@dataclass
class MyArgs:
    exp_name: str

    rl: RLArgs
    imitation: ImitationArgs
    eval: EvalArgs

    torch_deterministic: bool = True
    cuda: bool = True

    seed: Optional[int] = None  # only to be used for generated data

    # wandb
    wandb_project_name: str = "cleanRL"
    wandb_entity: str = None
    wandb_group: str = None
    wandb_tags: Optional[List] = None
