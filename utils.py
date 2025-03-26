import gymnasium as gym
import numpy as np


def visualize_rollout(env_name, act_function):
    env_render = gym.make(env_name, render_mode='human')
    obs, info = env_render.reset()
    done = False
    total_rew = 0
    while not done:
        act = act_function(obs)
        obs, rew, done, truncated, info = env_render.step(act)
        done = done or truncated
        total_rew += rew
    env_render.close()
