import sys
import os
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt

mypath = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
print("adding", mypath, "to the sys path")
sys.path.append(mypath)

from envs.ycb_env import Env


env = gym.make(
    id="Bimanual_Allegro_YCB",
    render_mode="human",
    control_mode="pd_joint_delta_pos",
    obs_mode="state",
    sim_backend="gpu",
)
env.reset()
img = env.render()

obs, _ = env.reset(seed=0)  # reset with a seed for determinism
done = False
while not done:
    action = env.action_space.sample()
    #action = np.zeros_like(action)  # replace with your own action
    obs, reward, terminated, truncated, info = env.step(action) # comment this out to freeze the simulation
    #done = terminated or truncated
    img = env.render()  # a display is required to render

env.close()
