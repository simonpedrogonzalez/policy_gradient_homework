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

def train(env_name='CartPole-v1', hidden_sizes=(32,), lr=1e-2, epochs=50, batch_size=5000, to_go=True, discount=1.0, lam=0.95, visualize=False, v_batch_size=1000):
    env = gym.make(env_name)
    ac = MLPActorCritic(env.observation_space, env.action_space, hidden_sizes=hidden_sizes)
    optimizer = Adam(ac.pi.parameters(), lr=lr)
    optimizer_v = Adam(ac.v.parameters(), lr=lr)
    loss_v = torch.nn.MSELoss()

    def train_one_epoch():
        batch_obs = []
        # batch_obs_prime = []
        batch_acts = []
        batch_weights = []
        batch_rets = []
        batch_returns = []
        batch_lens = []
        batch_rews = []
        batch_adv = []

        obs, info = env.reset()
        done, ep_rews = False, []
        ep_s, ep_s_prime = [], []

        subbatch_index = 0

        while True:
            batch_obs.append(obs.copy())
            ep_s.append(obs.copy())
            act, _, logp = ac.step(obs)

            obs, rew, terminated, truncated, info = env.step(act)
            done = terminated or truncated
            # batch_obs_prime.append(obs.copy())
            ep_s_prime.append(obs.copy())
            batch_rews.append(rew)

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


                # Compute the advantage
                with torch.no_grad():
                    Vs = ac.v(torch.as_tensor(np.array(ep_s), dtype=torch.float32))
                    Vs_prime = ac.v(torch.as_tensor(np.array(ep_s_prime), dtype=torch.float32))
                ep_rews_tensor = torch.as_tensor(np.array(ep_rews), dtype=torch.float32)
                deltas = ep_rews_tensor + discount * Vs_prime - Vs
                adv = discount_cumsum(deltas.numpy(), discount=discount*lam)
                batch_adv += list(adv)
                batch_returns += list(adv + Vs.squeeze().numpy())

                # Train V_pi
                if len(batch_obs) - subbatch_index > v_batch_size or len(batch_obs) > batch_size:
                    # take the subbatch not yet used to update V

                    subbatch_obs = batch_obs[subbatch_index:]
                    
                    # target
                    subbatch_weights = batch_weights[subbatch_index:]
                    # subbatch_weights = batch_returns[subbatch_index:]

                    subbatch_index = len(batch_obs)

                    subbatch_obs_tensor = torch.as_tensor(np.array(subbatch_obs), dtype=torch.float32)
                    subbatch_weights_tensor = torch.as_tensor(subbatch_weights, dtype=torch.float32)
                    v = ac.v(subbatch_obs_tensor)
                    loss_ = loss_v(v, subbatch_weights_tensor)
                    optimizer_v.zero_grad()
                    loss_.backward()
                    optimizer_v.step()

                    print(f'V_pi loss: {loss_.item()}')



                # Reset
                obs, info = env.reset()
                done, ep_rews = False, []
                ep_s, ep_s_prime = [], []

                if len(batch_obs) > batch_size:
                    break

        obs_tensor = torch.as_tensor(np.array(batch_obs), dtype=torch.float32)

        if isinstance(env.action_space, gym.spaces.Discrete):
            act_tensor = torch.as_tensor(np.array(batch_acts), dtype=torch.int32)
        elif isinstance(env.action_space, gym.spaces.Box):
            act_tensor = torch.as_tensor(np.array(batch_acts), dtype=torch.float32)


        obs_tensor = torch.as_tensor(np.array(batch_obs), dtype=torch.float32)
        adv_tensor = torch.as_tensor(batch_adv, dtype=torch.float32)
        # normalize the advantage
        adv_tensor = (adv_tensor - adv_tensor.mean()) / (adv_tensor.std() + 1e-8)

        logps_tensor = ac.get_logps(obs_tensor, act_tensor)
        loss = -(logps_tensor * adv_tensor).mean()
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
    epochs = list(range(50))
    rs = []
    for _ in range(5):
        rets = train(
            env_name='CartPole-v1',
            hidden_sizes=(32,),
            lr=1e-2,
            epochs=50,
            batch_size=5000,
            v_batch_size=500 # This should show 5 updates per epoch
        )
        # sns.lineplot(x=epochs, y=rets)
        # plt.xlabel('Epochs')
        # plt.ylabel('Average Return')
        # plt.savefig('advantage_cartpole.png')
        rs.append(rets)
    df = pd.DataFrame({
        'returns': [r for rs_ in rs for r in rs_],
        'epochs': epochs * 5,
        'method': 'GAE'
    })

    df.to_csv('advantage_cartpole.csv')

    # load rewards_to_go_vs_vanilla.csv
    df2 = pd.read_csv('reward_to_go_vs_vanilla.csv')
    # join the two dataframes
    df = pd.concat([df, df2])

    sns.lineplot(x='epochs', y='returns', hue='method', data=df)
    plt.xlabel('Epochs')
    plt.ylabel('Returns')
    plt.tight_layout()
    plt.savefig('advantage_cartpole.png')


    # Hopper-v5
    # rets = train(
    #     env_name='Hopper-v5',
    #     hidden_sizes=(64, 64),
    #     lr=1e-3,
    #     epochs=300,
    #     batch_size=5000,
    #     visualize=True
    # )

    # epochs = list(range(len(rets)))
    # sns.lineplot(x=epochs, y=rets)
    # plt.xlabel('Epochs')
    # plt.ylabel('Average Return')
    # plt.savefig('hopper_v5.png')

