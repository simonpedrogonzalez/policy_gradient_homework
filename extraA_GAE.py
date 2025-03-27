import torch
import numpy as np
from torch.optim import Adam
import gymnasium as gym
from core import MLPActorCritic, discount_cumsum
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import time
from utils import visualize_rollout

def train(
    env_name='CartPole-v1',
    hidden_sizes=(32,),
    lr=1e-2, lrv=1e-2,
    epochs=50, 
    batch_size=5000,
    v_iter=80,
    discount=0.99, lam=0.95,
    visualize=False):

    env = gym.make(env_name)
    ac = MLPActorCritic(env.observation_space, env.action_space, hidden_sizes=hidden_sizes)
    optimizer = Adam(ac.pi.parameters(), lr=lr)
    optimizer_v = Adam(ac.v.parameters(), lr=lrv)
    loss_v_fn = torch.nn.MSELoss()

    def train_one_epoch():
        batch_obs = []
        batch_acts = []
        batch_rets = []
        batch_advs = []

        obs, info = env.reset()
        done = False

        # These are just for logging
        ep_ret, ep_len = 0, 0
        ep_rets, ep_lens = [], []


        ep_values = []
        ep_rews = []

        while True:
            
            act, v, logp = ac.step(obs)
            obs_, r, done, truncated, _ = env.step(act)

            ep_ret += r
            ep_len += 1

            batch_obs.append(obs.copy())
            batch_acts.append(act)
            
            ep_rews.append(r)
            ep_values.append(v)
            
            obs = obs_

            if done or truncated:
                
                # Compute advantage
                ep_deltas = np.array(ep_rews) + \
                    discount * np.array(ep_values[1:] + [0]) - np.array(ep_values)
                
                ep_advs = discount_cumsum(ep_deltas, discount * lam)
                
                # Compute to_go returns
                ep_returns = discount_cumsum(ep_rews, discount)

                batch_advs += list(ep_advs)
                batch_rets += list(ep_returns)

                # Reset
                
                done, ep_rews, ep_values = False, [], []
                obs, info = env.reset()
                
                ep_lens.append(ep_len)
                ep_rets.append(ep_ret)
                ep_ret, ep_len = 0, 0
                
                if len(batch_obs) > batch_size:
                    break

        # Normalize advantages
        batch_advs = np.array(batch_advs)
        batch_advs = (batch_advs - batch_advs.mean()) / (batch_advs.std() + 1e-8)
        adv_tensor = torch.as_tensor(batch_advs, dtype=torch.float32)

        # Get log probabilities
        if isinstance(env.action_space, gym.spaces.Discrete):
            act_tensor = torch.as_tensor(np.array(batch_acts), dtype=torch.int32)
        elif isinstance(env.action_space, gym.spaces.Box):
            act_tensor = torch.as_tensor(np.array(batch_acts), dtype=torch.float32)
        obs_tensor = torch.as_tensor(np.array(batch_obs), dtype=torch.float32)

        _, logps_tensor = ac.pi(obs_tensor, act_tensor)

        # Compute Actor Loss
        loss = -(logps_tensor * adv_tensor).mean()

        # Update the Actor
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Update the Critic
        loss_v_acc = 0
        R = torch.as_tensor(np.array(batch_rets), dtype=torch.float32)
        for _ in range(v_iter):
            Vs = ac.v(obs_tensor)
            loss_v = loss_v_fn(Vs, R)
            optimizer_v.zero_grad()
            loss_v.backward()
            optimizer_v.step()
            loss_v_acc += loss_v.item()
        mean_loss_v = loss_v_acc / v_iter

        return ep_rets, loss.item(), mean_loss_v, ep_lens

    all_returns = []
    for i in range(epochs):
        t0 = time.time()
        rets, loss_pi, loss_v, lens = train_one_epoch()
        t1 = time.time()
        dt_s = t1 - t0
        print(f'Epoch {i:>2d} \t Return: {np.mean(rets):.3f} \t PLoss: {loss_pi:.3f} \t VLoss: {loss_v:.3f} \t EpLen: {np.mean(lens):.2f} \t Time: {dt_s:.2f}')
        all_returns.append(np.mean(rets))

    if visualize:
        for _ in range(5):
            input('Press Enter to visualize a rollout')
            visualize_rollout(env_name, lambda obs: ac.act(obs))

    env.close()
    return all_returns

if __name__ == '__main__':

    # Replication of CartPole-v1 results
    epochs = list(range(50))
    rs = []
    for _ in range(5):
        rets = train(
            env_name='CartPole-v1'
        )

        rs.append(rets)
    df = pd.DataFrame({
        'returns': [r for rs_ in rs for r in rs_],
        'epochs': epochs * 5,
        'method': 'GAE'
    })

    df.to_csv('advantage_cartpole.csv')

    df2 = pd.read_csv('reward_to_go_vs_vanilla.csv')
    df = pd.concat([df, df2])

    sns.lineplot(x='epochs', y='returns', hue='method', data=df)
    plt.xlabel('Epochs')
    plt.ylabel('Returns')
    plt.tight_layout()
    plt.savefig('advantage_cartpole.png')


    
    # Hopper-v5
    
    rs = []
    for _ in range(5):
        rets = train(
            env_name='Hopper-v5',
            hidden_sizes=(64, 64),
            epochs=500,
            batch_size=5000,
            visualize=False,
            lr=1e-3,
            lrv=1e-2,
            v_iter=80
        )
        rs.append(rets)

    epochs = list(range(len(rets)))

    df = pd.DataFrame({
        'returns': [r for rs_ in rs for r in rs_],
        'epochs': epochs * 5,
        'method': 'GAE'
    })
    
    df2 = pd.read_csv('hopper_v5_reward_to_go.csv')
    df = pd.concat([df, df2])
    df.to_csv('advantage_hopper_v5_2nd_test.csv')

    sns.lineplot(x='epochs', y='returns', hue='method', data=df)
    plt.xlabel('Epochs')
    plt.ylabel('Returns')
    plt.tight_layout()
    plt.savefig('advantage_hopper_v5_2nd_test.png')

