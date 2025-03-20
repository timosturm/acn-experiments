import optuna
import wandb
from uuid import uuid4
from torch.utils.tensorboard import SummaryWriter


from copy import deepcopy

from optuna import TrialPruned

import random

import gymnasium as gym
import torch
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from icecream import ic
from src.imitation.args import MyArgs
from imitation import imitate
from rl import train_ppo
from utils import save_state_dict, validate_on_env, clean_up


def objective_combined(
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
    args.rl.num_steps = trial.suggest_int(
        "num_steps / batch_size", 128, 2048, log=True)

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
        save_state_dict(args, run_name, state_dict, "imitation",
                        epoch, metadata=metadata)

        if new_return > old_return:
            best_return = new_return
            best_state_dict = state_dict
            save_state_dict(args, run_name, state_dict, "imitation",
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
        save_state_dict(args, run_name, state_dict, "rl",
                        global_step, metadata=metadata)

        if new_return > old_return:
            best_return = new_return
            best_state_dict = state_dict
            save_state_dict(args, run_name, state_dict, "rl",
                            i="best", metadata=metadata)

        trial.report(new_return, i)
        if trial.should_prune():
            clean_up(args, writer)
            raise TrialPruned()

        old_return = new_return

    clean_up(args, writer, wandb.run)

    return best_return


def objective_IL(
    trial: optuna.trial.Trial,
    args: MyArgs,
    make_env,  # TODO Typing
) -> float:
    args = deepcopy(args)  # because arguments are copy-by-reference

    args.imitation.lr = trial.suggest_float(
        "lr_imitation", 1e-5, 1e-2, log=True)
    # args.imitation.n_epochs = trial.suggest_categorical(
    #     "n_epochs_imitation", [5, 10, 20, 30, 40, 60, 80, 100])
    args.imitation.n_epochs = 1000

    # architecture optimization:
    n_layers = trial.suggest_int("n_hidden_layers", 2, 5)
    hiddens = [
        trial.suggest_int(f"n_neurons_layer_{i}", 64, 2048, log=True)
        for i in range(n_layers)
    ]
    args.imitation.hiddens = hiddens
    args.rl.hiddens = hiddens
    args.eval.hiddens = hiddens

    # args.rl.lr = trial.suggest_float(
    #     "lr_rl", 3e-5, 3e-3, log=True)
    # args.rl.ent_coef = trial.suggest_float("ent_coef_rl", 1e-6, 1e-2, log=True)
    # args.rl.gamma = trial.suggest_float("gamma_rl", 0.9, 0.99)
    # args.rl.gae_lambda = trial.suggest_float("gae_lambda_rl", 0, 1)
    # args.rl.max_grad_norm = trial.suggest_float("max_grad_norm_rl", 0.3, 0.7)
    # args.rl.vf_coef = trial.suggest_float("vf_coef_rl", 0, 1)
    # args.rl.clip_coef = trial.suggest_float("clip_coef_rl", 0, 1)
    # args.rl.num_steps = trial.suggest_int(
    #     "num_steps / batch_size", 128, 2048, log=True)

    # args.rl.batch_size = int(args.rl.num_envs * args.rl.num_steps)
    # args.rl.minibatch_size = int(args.rl.batch_size // args.rl.num_minibatches)
    # args.rl.num_iterations = args.rl.total_timesteps // args.rl.batch_size

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
    old_return = -np.inf
    best_return = -np.inf

    metadata = {
        "parameter": trial.params
    }

    for epoch, state_dict in imitate(args.imitation, writer, device, state_dict=None):

        eval_sim, new_return = validate_on_env(args.eval, state_dict)
        writer.add_scalar("imitation/return", new_return, epoch)
        
        print(f"epoch={epoch}, return={new_return}")

        for metric_name, f in args.eval.metrics.items():
            writer.add_scalar(
                f"imitation/{metric_name}", f(eval_sim), epoch)

        metadata |= {"imitation": {"return": new_return, "epoch": epoch}}
        save_state_dict(args, run_name, state_dict, "imitation",
                        epoch, metadata=metadata)

        if new_return > old_return:
            print(f"New best model at epoch={epoch} with return={new_return}!")
            best_return = new_return
            save_state_dict(args, run_name, state_dict, "imitation",
                            i="best", metadata=metadata)

        old_return = new_return

    clean_up(args, writer, wandb.run)

    return best_return
