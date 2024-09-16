from ray.rllib.algorithms import Algorithm
from tqdm import tqdm

from experiments.ppo_config import ppo_csc_config
from plotting import plot_training

try:
    algo = Algorithm.from_checkpoint("PPO_csc_check")
    print("Resuming from checkpoint")
except Exception:
    config = ppo_csc_config
    algo = config.build()
    print("Build new algorithm")

res = []

for i in tqdm(range(100), desc='training iteration'):
    res.append(algo.train())

algo.save(checkpoint_dir=f"PPO_csc_check")
plot_training(res).savefig(f"{algo}_csc.png", dpi=600)
