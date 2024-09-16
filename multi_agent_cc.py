from ray.rllib.algorithms import Algorithm
from tqdm import tqdm

from experiments.ppo_config import ppo_cc_config
from plotting import plot_training

try:
    algo = Algorithm.from_checkpoint("PPO_cc_check")
    print("Resuming from checkpoint")
except Exception:
    config = ppo_cc_config
    algo = config.build()
    print("Build new algorithm")

res = []

for i in tqdm(range(100), desc='training iteration'):
    res.append(algo.train())

algo.save(checkpoint_dir=f"PPO_cc_check")
plot_training(res).savefig(f"{algo}_cc.png", dpi=600)
