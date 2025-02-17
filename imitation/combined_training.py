import json
from optuna.pruners import MedianPruner
from src.cleanRL.environment import make_env
from src.pv.metrics import *
from gymportal.evaluation import *
from src.pv.rewards import *
from src.pv.observations import pv_observation_mean
from gymportal.environment import *
from src.pv.pv import read_pv_data
import pytz
from datetime import datetime, timedelta
from gymportal.sim import get_charging_network, Recomputer, EvaluationSimulator, SimGenerator
from gymportal.data.battery_generators import CustomizableBatteryGenerator
from acnportal.acnsim import Linear2StageBattery
from gymportal.data.ev_generators import get_standard_generator, RealWorldGenerator
import sys
import os
import optuna
import wandb
from uuid import uuid4
from torch.utils.tensorboard import SummaryWriter


from copy import deepcopy
import glob
from typing import Any, Callable, Dict, Generator, List, Optional, OrderedDict, Tuple, Type

from optuna import TrialPruned

from src.cleanRL.scheduler import CleanRLSchedule
from src.cleanRL.args import Args
from src.cleanRL.agent import Agent
from src.utils import evaluate_model
import random
import time

import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from icecream import ic
from src.imitation.args import MyArgs, ImitationArgs, RLArgs, EvalArgs
from src.imitation.dataset import MyDataset, TransformAction
from torch.utils.data import DataLoader
from src.imitation.utils import collate_to_float32


def _save_state_dict(run_name: str, state_dict: OrderedDict, type: str, i: int, metadata: Optional[Dict[str, Any]] = None):
    path = f"runs/{run_name}/models/{type}/"
    name = f"{args.exp_name}_{i}"

    from pathlib import Path
    Path(path).mkdir(parents=True, exist_ok=True)

    torch.save(state_dict, f"{path}/{name}.mdl")

    artifact = wandb.Artifact(
        f'model_{type}{i}',
        type=f'model_{type}',
        metadata=metadata,
    )
    artifact.add_file(f"{path}/{name}.mdl")
    wandb.log_artifact(artifact)

    with open(f"{path}/{name}.json", 'w') as file:
        file.write(json.dumps(metadata))


def validate_on_env(args: EvalArgs, state_dict: OrderedDict) -> float:
    agent = Agent(
        observation_shape=np.array(args.env.observation_space.shape).prod(),
        action_shape=np.array(args.env.action_space.shape).prod()
    )

    agent.load_state_dict(state_dict)
    scheduler = CleanRLSchedule(agent)

    eval_sim, new_return = evaluate_model(
        scheduler,
        args.env,
        seed=args.env.unwrapped.simgenerator.seed
    )

    return eval_sim, new_return


def imitate(
    args: ImitationArgs,
    writer: SummaryWriter,
    device: str,
    state_dict: Optional[OrderedDict],
) -> Generator[Tuple[int, OrderedDict], None, None]:

    train_ds = MyDataset(args.train_ds, transform=TransformAction())
    validation_ds = MyDataset(args.validation_ds, transform=TransformAction())

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_to_float32,
    )

    validation_loader = DataLoader(
        validation_ds,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_to_float32,
    )

    agent = Agent(
        observation_shape=np.prod(train_ds[0]["observation"].shape),
        action_shape=np.prod(train_ds[0]["action"].shape)
    ).to(device)

    if state_dict:
        agent.load_state_dict(state_dict)

    optimizer = optim.Adam(agent.parameters(), lr=args.lr)

    criterion = nn.MSELoss()

    def calc_val_loss(agent, validation_loader):
        actions = []
        true_actions = []

        for x in validation_loader:
            action, log_prob, entropy, value = agent.get_action_and_value(
                x["observation"])

            actions.append(action)
            true_actions.append(x["action"])

        loss = criterion(torch.vstack(actions), torch.vstack(true_actions))

        return loss

    for epoch in range(args.n_epochs):
        for step, x in enumerate(train_loader):
            action, _, _, _ = agent.get_action_and_value(x["observation"])

            loss = criterion(action, x["action"])
            writer.add_scalar("imitation/train_loss", loss.item(), epoch)

            if step % 100 == 0:
                agent.eval()
                val_loss = calc_val_loss(agent, validation_loader)
                writer.add_scalar("imitation/validation_loss",
                                  val_loss.item(), epoch)
                agent.train()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        yield epoch, agent.state_dict()


def train_ppo(
    args: RLArgs,
    writer: SummaryWriter,
    device: str,
    state_dict: Optional[OrderedDict],
) -> Generator[Tuple[int, OrderedDict], None, None]:
    envs = args.envs

    assert isinstance(envs.single_action_space,
                      gym.spaces.Box), "only continuous action space is supported"

    agent = Agent(
        observation_shape=np.array(envs.single_observation_space.shape).prod(),
        action_shape=np.array(envs.single_action_space.shape).prod()
    ).to(device)

    if state_dict:
        agent.load_state_dict(state_dict)

    optimizer = optim.Adam(agent.parameters(), lr=args.lr)

    # ALGO Logic: Storage setup
    obs = torch.zeros((args.num_steps, args.num_envs) +
                      envs.single_observation_space.shape).to(device)
    actions = torch.zeros((args.num_steps, args.num_envs) +
                          envs.single_action_space.shape).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    start_time = time.time()
    next_obs, _ = envs.reset()  # envs.reset(seed=args.seed)
    next_obs = torch.Tensor(next_obs).to(device)
    next_done = torch.zeros(args.num_envs).to(device)

    for iteration in range(1, args.num_iterations + 1):
        # Annealing the rate if instructed to do so.
        if args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            lrnow = frac * args.lr
            optimizer.param_groups[0]["lr"] = lrnow

        # collect trajectories
        for step in range(0, args.num_steps):
            global_step += args.num_envs
            obs[step] = next_obs
            dones[step] = next_done

            # ALGO LOGIC: action logic
            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(
                    next_obs)
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob

            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs, reward, terminations, truncations, infos = envs.step(
                action.cpu().numpy())
            next_done = np.logical_or(terminations, truncations)
            rewards[step] = torch.tensor(reward).to(device).view(-1)
            next_obs, next_done = torch.Tensor(next_obs).to(
                device), torch.Tensor(next_done).to(device)

            if "final_info" in infos:
                for info in infos["final_info"]:
                    if info and "episode" in info:
                        print(
                            f"global_step={global_step}, episodic_return={info['episode']['r']}")
                        writer.add_scalar("charts/episodic_return",
                                          info["episode"]["r"], global_step)
                        writer.add_scalar("charts/episodic_length",
                                          info["episode"]["l"], global_step)

            # log metrics for training
            if torch.any(next_done):
                sims = []
                for info in infos["final_info"]:
                    if info and "acn_interface" in info:
                        sims.append(info["acn_interface"]._simulator)

                for metric_name, f in args.metrics.items():
                    value = np.mean([f(sim) for sim in sims])
                    writer.add_scalar(
                        f"train/{metric_name}", value, global_step)

        # bootstrap value if not done
        with torch.no_grad():
            next_value = agent.get_value(next_obs).reshape(1, -1)
            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                delta = rewards[t] + args.gamma * \
                    nextvalues * nextnonterminal - values[t]
                advantages[t] = lastgaelam = delta + args.gamma * \
                    args.gae_lambda * nextnonterminal * lastgaelam
            returns = advantages + values

        # flatten the batch
        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        # Optimizing the policy and value network
        b_inds = np.arange(args.batch_size)
        clipfracs = []
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue = agent.get_action_and_value(
                    b_obs[mb_inds], b_actions[mb_inds])
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() >
                                   args.clip_coef).float().mean().item()]

                mb_advantages = b_advantages[mb_inds]
                if args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()
                                     ) / (mb_advantages.std() + 1e-8)

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * \
                    torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                newvalue = newvalue.view(-1)
                if args.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -args.clip_coef,
                        args.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * \
                        ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    agent.parameters(), args.max_grad_norm)
                optimizer.step()

            if args.target_kl is not None and approx_kl > args.target_kl:
                break

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - \
            np.var(y_true - y_pred) / var_y

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/old_approx_kl",
                          old_approx_kl.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        writer.add_scalar("losses/explained_variance",
                          explained_var, global_step)
        print("SPS:", int(global_step / (time.time() - start_time)))
        writer.add_scalar("charts/SPS", int(global_step /
                                            (time.time() - start_time)), global_step)

        yield global_step, agent.state_dict()


def objective(
    trial: optuna.trial.Trial,
    args: MyArgs,
    make_env,  # TODO Typing
) -> float:
    args = deepcopy(args)  # because arguments are copy-by-reference

    args.imitation.lr = trial.suggest_float(
        "lr_imitation", 1e-5, 1e-2, log=True)
    args.imitation.n_epochs = trial.suggest_categorical(
        "n_epochs_imitation", [5, 10, 20, 30, 40])

    args.rl.lr = trial.suggest_float(
        "lr_rl", 3e-5, 3e-3, log=True)
    args.rl.ent_coef = trial.suggest_float("ent_coef_rl", 1e-6, 1e-2, log=True)
    args.rl.gamma = trial.suggest_float("gamma_rl", 0.9, 0.99)
    args.rl.gae_lambda = trial.suggest_float("gae_lambda_rl", 0, 1)
    args.rl.max_grad_norm = trial.suggest_float("max_grad_norm_rl", 0.3, 0.7)
    args.rl.vf_coef = trial.suggest_float("vf_coef_rl", 0, 1)
    args.rl.clip_coef = trial.suggest_float("clip_coef_rl", 0, 1)
    args.rl.num_steps = trial.suggest_float("num_steps / batch_size", 128, 2048, log=True)

    args.rl.batch_size = int(args.rl.num_envs * args.rl.num_steps)
    args.rl.minibatch_size = int(args.rl.batch_size // args.rl.num_minibatches)
    args.rl.num_iterations = args.rl.total_timesteps // args.rl.batch_size

    args.rl.envs = gym.vector.SyncVectorEnv(
        [make_env(args.rl.config, args.rl.gamma, i)
         for i in range(args.rl.num_envs)]
    )

    args.wandb_tags = [f"{key}={value}" for key, value in trial.params.items()]

    run_name = f"{args.exp_name}__{uuid4()}"

    wandb.init(
        project=args.wandb_project_name,
        entity=args.wandb_entity,
        sync_tensorboard=True,
        config=trial.params,  # vars(args),
        name=run_name,
        save_code=True,
        group=args.wandb_group,
        tags=args.wandb_tags,
    )

    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % (
            "\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available()
                          and args.cuda else "cpu")

    # Imitatation Learning
    best_state_dict = None
    old_return = -np.inf
    best_return = -np.inf

    metadata = {
        "parameter": trial.params
    }

    for epoch, state_dict in imitate(args.imitation, writer, device, state_dict=None):

        eval_sim, new_return = validate_on_env(args.eval, state_dict)
        writer.add_scalar("imitation/return", new_return, epoch)

        for metric_name, f in args.eval.metrics.items():
            writer.add_scalar(
                f"imitation/{metric_name}", f(eval_sim), epoch)

        metadata |= {"imitation": {"return": new_return, "epoch": epoch}}
        _save_state_dict(run_name, state_dict, "imitation",
                         epoch, metadata=metadata)

        if new_return > old_return:
            best_return = new_return
            best_state_dict = state_dict
            _save_state_dict(run_name, state_dict, "imitation",
                             i="best", metadata=metadata)

        old_return = new_return

    assert best_state_dict, "Something went wrong, there is no best_state_dict!"

    old_return = 0
    # Reinforcement Learning
    for i, (global_step, state_dict) in enumerate(train_ppo(args.rl, writer, device, state_dict=best_state_dict)):
        eval_sim, new_return = validate_on_env(args.eval, state_dict)
        writer.add_scalar("eval/return", new_return, global_step)

        for metric_name, f in args.eval.metrics.items():
            writer.add_scalar(f"eval/{metric_name}", f(eval_sim), global_step)

        metadata |= {"rl": {"return": new_return, "global_step": global_step}}
        _save_state_dict(run_name, state_dict, "rl",
                         global_step, metadata=metadata)

        if new_return > old_return:
            best_return = new_return
            best_state_dict = state_dict
            _save_state_dict(run_name, state_dict, "rl",
                             i="best", metadata=metadata)

        trial.report(new_return, i)
        if trial.should_prune():
            _clean_up(args, writer)
            raise TrialPruned()

        old_return = new_return

    _clean_up(args, writer, wandb.run)

    return best_return


def _clean_up(args: MyArgs, writer: SummaryWriter, run):

    args.rl.envs.close()
    args.eval.env.close()

    writer.flush()
    writer.close()

    run.finish()


# This is importent when we want to call this as a python script, because jupyter naturally has a higher recursion depth
sys.setrecursionlimit(3000)

# Print the PID when using nohup
ic(os.getpid())


timezone = pytz.timezone("America/Los_Angeles")


# charging_network = get_charging_network('simple_acn', basic_evse=True, voltage=208,
#                                         network_kwargs={
#                                             'station_ids': ['CA-504', 'CA-503', 'CA-502', 'CA-501'],
#                                             # 'station_ids': ['CA-501'],
#                                             "aggregate_cap": 32 * 208 / 1000})

charging_network = get_charging_network('caltech', basic_evse=True, voltage=208,
                                        network_kwargs={"transformer_cap": 150})

battery_generator = CustomizableBatteryGenerator(
    voltage=208,
    period=1,
    battery_types=[
        Linear2StageBattery],
    max_power_function="normal",
)

# ev_generator = RealWorldGenerator(battery_generator=battery_generator, site='caltech', period=1)
ev_generator = get_standard_generator(
    'caltech', battery_generator, seed=42, frequency_multiplicator=10, duration_multiplicator=2)

# TODO Use time intervals and GMMs from https://github.com/chrisyeh96/sustaingym/blob/main/sustaingym/envs/evcharging/utils.py
# I.e., train on generated data, evaluate on new generated data and real data from the same interval
# optional: compare to "out-of-distribution" data from different interval

train_generator = SimGenerator(
    charging_network=charging_network,
    simulation_days=1,
    n_intervals=7 * 46,
    start_date=timezone.localize(datetime(2019, 1, 1)),
    ev_generator=ev_generator,
    recomputer=Recomputer(recompute_interval=10, sparse=True),
    sim_class=EvaluationSimulator,
)

ic(train_generator.end_date + timedelta(days=1))

validation_generator = SimGenerator(
    charging_network=charging_network,
    simulation_days=7,
    n_intervals=1,
    start_date=train_generator.end_date + timedelta(days=1),
    ev_generator=ev_generator,
    recomputer=Recomputer(recompute_interval=10, sparse=True),
    sim_class=EvaluationSimulator,
)

ic(validation_generator.end_date + timedelta(days=1))

test_generator = SimGenerator(
    charging_network=charging_network,
    simulation_days=14,
    n_intervals=1,
    start_date=validation_generator.end_date + timedelta(days=1),
    ev_generator=ev_generator,
    recomputer=Recomputer(recompute_interval=10, sparse=True),
    sim_class=EvaluationSimulator,
)

ic(test_generator.end_date + timedelta(days=1))


df_pv = read_pv_data("../pv_150kW.csv")
df_pv.describe()


observation_objects = [
    charging_rates_observation_normalized(),
    percentage_of_magnitude_observation(),
    diff_pilots_charging_rates_observation_normalized(),
    cyclical_minute_observation(),
    cyclical_day_observation(),
    cyclical_month_observation(),
    cyclical_minute_observation_stay(),
    energy_delivered_observation_normalized(),
    num_active_stations_observation_normalized(),
    pilot_signals_observation_normalized(),
    pv_observation_mean(df_pv),
]

reward_objects = [
    pv_utilization_reward(df_pv),
    # grid_use_penalty(df_pv),
    unused_pv_penalty(df_pv),
    charging_reward(),
    # soft_charging_reward_pv_weighted(df_pv, transformer_cap=150),
]

train_generator.seed = 8734956
_ = train_generator.reset()

iter = 0

while train_generator._current_date != train_generator.start_date:
    _ = train_generator.next()

    ic(iter)
    ic(train_generator._current_date)
    iter += 1

steps_per_epoch = 0
for eval_sim in train_generator._sim_memory:
    steps_per_epoch += len(eval_sim.event_queue.queue)

ic(steps_per_epoch)

train_config = {"observation_objects": observation_objects, "action_object": zero_centered_single_charging_schedule_normalized(),
                "reward_objects": reward_objects,
                "simgenerator": train_generator,
                "meet_constraints": True}

validation_config = train_config | {'simgenerator': validation_generator}
test_config = train_config | {'simgenerator': test_generator}

metrics = {
    "SoC >= 90%": percentage_soc,
    "mean SoC": mean_soc,
    "median SoC": median_soc,
    # "prop feasible steps": proportion_of_feasible_charging,
    # "prop feasible charge": proportion_of_feasible_charge,
    "pv utilization": lambda sim: pv_utilization_metric(sim, df_pv),
    "grid usage": lambda sim: grid_use_metric(sim, df_pv),
    "unused pv": lambda sim: unused_pv_metric(sim, df_pv),
}

args = MyArgs(
    exp_name="Imitation+RL",
    wandb_project_name="imitation+rl",
    wandb_group="test run",
    seed=42,  # TODO
    imitation=ImitationArgs(
        # TODO Store baseline as a parameter
        train_ds="FCFS_gen_training.parquet.gzip",
        validation_ds="FCFS_gen_validation.parquet.gzip",
    ),
    eval=EvalArgs(
        make_env=lambda: make_env(validation_config, 0.99, 0, 930932)(),
        metrics=metrics,
    ),
    rl=RLArgs(
        total_timesteps=steps_per_epoch * 20,
        config=train_config,
        metrics=metrics,
    ),
)

if __name__ == "__main__":
    study = optuna.create_study(
        study_name='fine-tuning',
        storage='sqlite:///fine-tuning.db',
        load_if_exists=True,
        direction="maximize",
        # sampler=
        pruner=MedianPruner(
            n_startup_trials=10,
            n_warmup_steps=5
        )
    )

    def objective_wrapper(trial):
        return objective(
            trial,
            args,
            make_env,
        )

    study.optimize(objective_wrapper, n_trials=1)  # TODO Add more trials

    # TODO Do something with the best model here (?)
