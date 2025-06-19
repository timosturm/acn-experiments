from torch.utils.tensorboard import SummaryWriter


from typing import Dict, Generator, Optional, OrderedDict, Tuple

from src.cleanRL.agent import Agent
import time

import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from icecream import ic
from src.imitation.args import RLArgs


def train_independent_ppo(
    args: RLArgs,
    writer: SummaryWriter,
    device: str,
    state_dicts: Optional[Dict[str, OrderedDict]],
) -> Generator[Tuple[int, Dict[str, OrderedDict]], None, None]:
    envs = args.envs
    # agent_ids = args.agent_ids
    # TODO agent_ids = ???
    agent_ids = None

    for agent_id in agent_ids:
        assert isinstance(
            envs.single_action_space[agent_id], gym.spaces.Box), "only continuous action space is supported for agent " + agent_id

    agents = {
        agent_id: args.agent_class(
            observation_shape=np.array(
                envs.single_observation_space[agent_id].shape).prod(),
            action_shape=np.array(
                envs.single_action_space[agent_id].shape).prod(),
            hiddens=args.hiddens,
        ).to(device) for agent_id in agent_ids
    }

    if state_dicts:
        for agent_id, state_dict in state_dicts.items():
            agents[agent_id].load_state_dict(state_dict)

    optimizers = {
        agent_id: optim.Adam(agent.parameters(), lr=args.lr)
        for agent_id, agent in agents.items()
    }

    # ALGO Logic: Storage setup
    storage = {}
    for agent_id in agent_ids:
        storage[agent_id] = {
            'obs': torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space[agent_id].shape).to(device),
            'actions': torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space[agent_id].shape).to(device),
            'logprobs': torch.zeros((args.num_steps, args.num_envs)).to(device),
            'rewards': torch.zeros((args.num_steps, args.num_envs)).to(device),
            'dones': torch.zeros((args.num_steps, args.num_envs)).to(device),
            'values': torch.zeros((args.num_steps, args.num_envs)).to(device),
        }

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    start_time = time.time()
    next_obs, _ = envs.reset()
    next_obs = {
        agent_id: torch.Tensor(next_obs[agent_id]).to(device)
        for agent_id in agent_ids
    }
    next_done = {
        agent_id: torch.zeros(args.num_envs).to(device)
        for agent_id in agent_ids
    }

    for iteration in range(1, args.num_iterations + 1):
        # Annealing the rate if instructed to do so.
        if args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            lrnow = frac * args.lr
            for opt in optimizers.values():
                opt.param_groups[0]["lr"] = lrnow

        # collect trajectories
        for step in range(0, args.num_steps):
            global_step += args.num_envs

            for agent_id in agent_ids:
                storage[agent_id]['obs'][step] = next_obs[agent_id]
                storage[agent_id]['dones'][step] = next_done[agent_id]

                with torch.no_grad():
                    agents[agent_id].eval()
                    action, logprob, _, value = agents[agent_id].get_action_and_value(
                        next_obs[agent_id])
                    storage[agent_id]['values'][step] = value.flatten()
                    agents[agent_id].train()

                storage[agent_id]['actions'][step] = action
                storage[agent_id]['logprobs'][step] = logprob

            # TRY NOT TO MODIFY: execute the game and log data.
            actions_cpu = {
                agent_id: storage[agent_id]['actions'][step].cpu().numpy()
                for agent_id in agent_ids
            }

            (
                next_obs,
                rewards,
                terminations,
                truncations,
                infos,
            ) = envs.step(actions_cpu)

            for agent_id in agent_ids:
                next_done[agent_id] = np.logical_or(
                    terminations[agent_id], 
                    truncations[agent_id]
                    )
                storage[agent_id]['rewards'][step] = torch.tensor(
                    rewards[agent_id]).to(device).view(-1)
                next_obs[agent_id] = torch.Tensor(
                    next_obs[agent_id]).to(device)
                next_done[agent_id] = torch.Tensor(
                    next_done[agent_id]).to(device)

        agent_states = {}
        for agent_id in agent_ids:
            agent = agents[agent_id]
            optimizer = optimizers[agent_id]
            obs, actions, logprobs, rewards, dones, values = (
                storage[agent_id]['obs'], storage[agent_id]['actions'], storage[agent_id]['logprobs'],
                storage[agent_id]['rewards'], storage[agent_id]['dones'], storage[agent_id]['values']
            )

            with torch.no_grad():
                agent.eval()
                next_value = agent.get_value(next_obs[agent_id]).reshape(1, -1)
                advantages = torch.zeros_like(rewards).to(device)
                lastgaelam = 0
                for t in reversed(range(args.num_steps)):
                    nextnonterminal = 1.0 - \
                        (next_done[agent_id] if t ==
                         args.num_steps - 1 else dones[t + 1])
                    nextvalues = next_value if t == args.num_steps - \
                        1 else values[t + 1]
                    delta = rewards[t] + args.gamma * \
                        nextvalues * nextnonterminal - values[t]
                    advantages[t] = lastgaelam = delta + args.gamma * \
                        args.gae_lambda * nextnonterminal * lastgaelam
                returns = advantages + values
                agent.train()

            b_obs = obs.reshape(
                (-1,) + envs.single_observation_space[agent_id].shape)
            b_logprobs = logprobs.reshape(-1)
            b_actions = actions.reshape(
                (-1,) + envs.single_action_space[agent_id].shape)
            b_advantages = advantages.reshape(-1)
            b_returns = returns.reshape(-1)
            b_values = values.reshape(-1)

            b_inds = np.arange(args.batch_size)
            for epoch in range(args.update_epochs):
                np.random.shuffle(b_inds)
                for start in range(0, args.batch_size, args.minibatch_size):
                    end = start + args.minibatch_size
                    mb_inds = b_inds[start:end]

                    _, newlogprob, entropy, newvalue = agent.get_action_and_value(
                        b_obs[mb_inds], b_actions[mb_inds])
                    logratio = newlogprob - b_logprobs[mb_inds]
                    ratio = logratio.exp()

                    mb_advantages = b_advantages[mb_inds]
                    if args.norm_adv:
                        mb_advantages = (
                            mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                    pg_loss1 = -mb_advantages * ratio
                    pg_loss2 = -mb_advantages * \
                        torch.clamp(ratio, 1 - args.clip_coef,
                                    1 + args.clip_coef)
                    pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                    newvalue = newvalue.view(-1)
                    if args.clip_vloss:
                        v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                        v_clipped = b_values[mb_inds] + torch.clamp(
                            newvalue - b_values[mb_inds], -args.clip_coef, args.clip_coef)
                        v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                        v_loss = 0.5 * \
                            torch.max(v_loss_unclipped, v_loss_clipped).mean()
                    else:
                        v_loss = 0.5 * \
                            ((newvalue - b_returns[mb_inds]) ** 2).mean()

                    entropy_loss = entropy.mean()
                    loss = pg_loss - args.ent_coef * entropy_loss + args.vf_coef * v_loss

                    optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(
                        agent.parameters(), args.max_grad_norm)
                    optimizer.step()

            agent_states[agent_id] = agent.state_dict()

        yield global_step, agent_states
