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

def train(env_name='CartPole-v1', hidden_sizes=(32,), lr=1e-2, epochs=50, batch_size=5000, to_go=True, discount=1.0, visualize=False):
    env = gym.make(env_name)
    ac = MLPActorCritic(env.observation_space, env.action_space, hidden_sizes=hidden_sizes)
    optimizer = Adam(ac.pi.parameters(), lr=lr)

    def train_one_epoch():
        batch_obs = []
        batch_acts = []
        batch_weights = []
        batch_rets = []
        batch_lens = []

        obs, info = env.reset()
        done, ep_rews = False, []

        while True:
            batch_obs.append(obs.copy())
            act, _, logp = ac.step(obs)

            obs, rew, terminated, truncated, info = env.step(act)
            done = terminated or truncated

            batch_acts.append(act)
            ep_rews.append(rew)

            if done:
                ep_ret = sum(ep_rews)
                batch_rets.append(ep_ret)
                batch_lens.append(len(ep_rews))

                weights = (
                    discount_cumsum(ep_rews, discount=discount) if to_go
                    else [ep_ret] * len(ep_rews)
                )
                batch_weights += list(weights)

                obs, info = env.reset()
                done, ep_rews = False, []

                if len(batch_obs) > batch_size:
                    break

        obs_tensor = torch.as_tensor(np.array(batch_obs), dtype=torch.float32)
        
        if isinstance(env.action_space, gym.spaces.Discrete):
            act_tensor = torch.as_tensor(np.array(batch_acts), dtype=torch.int32)
        elif isinstance(env.action_space, gym.spaces.Box):
            act_tensor = torch.as_tensor(np.array(batch_acts), dtype=torch.float32)
        
        weights_tensor = torch.as_tensor(batch_weights, dtype=torch.float32)
        
        logps_tensor = ac.get_logps(obs_tensor, act_tensor)

        loss = -(logps_tensor * weights_tensor).mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        return loss.item(), batch_rets, batch_lens

    all_returns = []
    for i in range(epochs):
        t0 = time.time()
        loss, rets, lens = train_one_epoch()
        t1 = time.time()
        dt_s = t1 - t0
        print(f'Epoch {i:>2d} \t Loss: {loss:.3f} \t Return: {np.mean(rets):.3f} \t EpLen: {np.mean(lens):.2f} \t Time: {dt_s:.2f}')
        all_returns.append(np.mean(rets))

    if visualize:
        for _ in range(5):
            input('Press Enter to visualize a rollout')
            visualize_rollout(env_name, lambda obs: ac.act(obs))

    env.close()
    return all_returns

if __name__ == '__main__':

    # Replication of CartPole-v1 results
    rets = train(
        env_name='CartPole-v1',
        hidden_sizes=(32,),
        lr=1e-2,
        epochs=50,
        batch_size=5000
    )
    epochs = list(range(len(rets)))
    sns.lineplot(x=epochs, y=rets)
    plt.xlabel('Epochs')
    plt.ylabel('Average Return')
    plt.savefig('replicate_cartpole.png')

    # Hopper-v5
    rs = []
    for _ in range(5):
        rets = train(
            env_name='Hopper-v5',
            hidden_sizes=(64, 64),
            lr=1e-3,
            epochs=500,
            batch_size=5000,
            visualize=False
        )
        rs.append(rets)


    epochs = list(range(len(rets)))
    df = pd.DataFrame({
        'returns': [r for rs_ in rs for r in rs_],
        'epochs': epochs * 5,
        'method': 'Reward to go'
    })

    df.to_csv('hopper_v5_reward_to_go.csv')

    sns.lineplot(x=epochs, y=rets)
    plt.xlabel('Epochs')
    plt.ylabel('Average Return')
    plt.savefig('hopper_v5.png')

