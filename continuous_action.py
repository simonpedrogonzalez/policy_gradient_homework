from core import MLPActorCritic, MLPGaussianActor, MLPCategoricalActor

import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical
from torch.optim import Adam
import numpy as np
import gymnasium as gym
from gymnasium.spaces import Discrete, Box
import seaborn as sns
import matplotlib.pyplot as plt

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")

def train(env, env_render=None, n_epochs=50):

    epochs = n_epochs

    ac = MLPActorCritic(
        env.observation_space,
        env.action_space,
        hidden_sizes=(32,),
    ).to(device)

    # make action selection function (outputs int actions, sampled from policy)
    def get_action(obs):
        return ac.pi.forward(obs).sample().item()

    # make loss function whose gradient, for the right data, is policy gradient
    def compute_loss(obs, act, weights, logps):
        # _, logps = ac.pi.forward(obs, act)
        return -(logps * weights).mean()

    # Let's see what is happening
    def visualize_one_rollout():
        obs, info = env_render.reset()
        done = False
        cumulative_rew = 0
        while not done:
            env_render.render()
            act = ac.act(torch.as_tensor(obs, dtype=torch.float32).to(device))
            obs, rew, done, truncated, info = env_render.step(act.cpu().numpy())
            done = done or truncated
            cumulative_rew += rew
        env.close()
        print(f'Rollout cumulative reward: {round(cumulative_rew, 3)}')

    # make optimizer
    if isinstance(ac.pi, MLPCategoricalActor):
        params = ac.pi.logits_net.parameters()
    elif isinstance(ac.pi, MLPGaussianActor):
        params = ac.pi.mu_net.parameters()
    else:
        raise ValueError('missing net parameters')
    
    lr=1e-2
    batch_size = 5000
    optimizer = Adam(params, lr=lr)

    # for training policy
    def train_one_epoch():
        
        def reward_to_go(rews):
            n = len(rews)
            rtgs = np.zeros_like(rews)
            for i in reversed(range(n)):
                rtgs[i] = rews[i] + (rtgs[i+1] if i+1 < n else 0)
            return rtgs

        # make some empty lists for logging.
        batch_obs = []          # for observations
        batch_acts = []         # for actions
        batch_weights = []      # for R(tau) weighting in policy gradient
        batch_rets = []         # for measuring episode returns
        batch_lens = []         # for measuring episode lengths
        logps = []

        # reset episode-specific variables
        obs, info = env.reset()  # updated reset handling
        done = False            # signal from environment that episode is over
        ep_rews = []            # list for rewards accrued throughout ep

        # render first episode of each epoch
        finished_rendering_this_epoch = False

        # collect experience by acting in the environment with current policy
        while True:

            # save obs
            batch_obs.append(obs.copy())

            # act in the environment
            act, _, logp_a = ac.step_grad(torch.as_tensor(obs, dtype=torch.float32).to(device))

            obs, rew, terminated, truncated, info = env.step(act.cpu().numpy())
            
            done = terminated or truncated

            # save action, reward
            batch_acts.append(act)
            logps.append(logp_a)

            ep_rews.append(rew)

            if done:
                # if episode is over, record info about episode
                ep_ret, ep_len = sum(ep_rews), len(ep_rews)
                batch_rets.append(ep_ret)
                batch_lens.append(ep_len)

                # reward to-go
                batch_weights += list(reward_to_go(ep_rews))

                # reset episode-specific variables
                obs, info = env.reset()
                done, ep_rews = False, []

                # end experience loop if we have enough of it
                if len(batch_obs) > batch_size:
                    break

        # take a single policy gradient update step
        optimizer.zero_grad()
        batch_loss = compute_loss(
            obs=torch.as_tensor(np.array(batch_obs), dtype=torch.float32).to(device),
            act=torch.stack(batch_acts),
            weights=torch.as_tensor(np.array(batch_weights), dtype=torch.float32).to(device),
            logps=torch.stack(logps)
        )
        batch_loss.backward()
        optimizer.step()
        return batch_loss, batch_rets, batch_lens

    epoch_rets = []
    # training loop
    for i in range(epochs):
        batch_loss, batch_rets, batch_lens = train_one_epoch()
        
        # Every 10 epochs otherwise it's difficult to
        # notice a difference
        # if i % 1 == 0:
        #     visualize_one_rollout()
        print('epoch: %3d \t loss: %.3f \t return: %.3f \t ep_len: %.3f'%
                (i, batch_loss, np.mean(batch_rets), np.mean(batch_lens)))
        epoch_rets.append(np.mean(batch_rets))

    # wait for me to press enter before visualizing the policy
    input('Press enter to visualize the policy in action ...')
    visualize_one_rollout()
    return epoch_rets

# CartPole

# env = gym.make('CartPole-v1')
# rets = train(env, n_epochs=50)
# epochs = list(range(1, len(rets)+1))
# sns.lineplot(x=epochs, y=rets)
# plt.xlabel('Epochs')
# plt.ylabel('Returns')
# plt.tight_layout()
# plt.savefig('test_pole.png')
# env.close()

# Hopper

# env = gym.make('Hopper-v5')
# env_render = gym.make('Hopper-v5', render_mode='human')
# rets = train(env, env_render, n_epochs=50)
# epochs = list(range(1, len(rets)+1))
# plt.figure(figsize=(4, 4))
# sns.lineplot(x=epochs, y=rets)
# plt.xlabel('Epochs')
# plt.ylabel('Returns')
# plt.tight_layout()
# plt.savefig('test_hopper.png')
# env.close()
