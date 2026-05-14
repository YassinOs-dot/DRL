#!/usr/bin/env python
"""Run a few episodes with random actions and plot the TCP path."""
import os
os.environ["MOCK_EGM"] = "1"          # force mock mode

import numpy as np
import matplotlib.pyplot as plt
from drl_welding.envs.welding_env import WeldingEnv
from drl_welding.utils.seam_utils import SeamGenerator
import yaml

def main():
    with open("config/default.yaml") as f:
        config = yaml.safe_load(f)
    config['seam_config'] = {'seam_type': 'line', 'start': [200,0,100], 'end': [600,0,100]}
    config['obstacles'] = []          # no obstacles for first test
    config['max_episode_steps'] = 200

    env = WeldingEnv(config)
    obs, _ = env.reset()

    tcp_positions = []
    rewards = []

    for _ in range(200):
        action = env.action_space.sample()   # random policy
        obs, reward, done, truncated, info = env.step(action)
        tcp_positions.append(env.egm.tcp_pos.copy())
        rewards.append(reward)
        if done or truncated:
            break

    tcp_positions = np.array(tcp_positions)

    # Plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(tcp_positions[:,0], tcp_positions[:,1], tcp_positions[:,2], 'b-', label='TCP path')
    # Plot the seam
    seam = SeamGenerator(config['seam_config'])
    t = np.linspace(0,1,100)
    seam_pts = seam.spline(t)
    ax.plot(seam_pts[:,0], seam_pts[:,1], seam_pts[:,2], 'r--', label='Seam')
    ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
    ax.legend()
    plt.title("Random agent trajectory (mock)")
    plt.show()

    print(f"Total reward: {sum(rewards):.1f}, steps: {len(tcp_positions)}")

if __name__ == "__main__":
    main()
