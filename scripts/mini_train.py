#!/usr/bin/env python
"""Train for a few thousand steps with mock EGM and log to TensorBoard."""
import os
os.environ["MOCK_EGM"] = "1"

import yaml
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from drl_welding.envs.welding_env import WeldingEnv

def main():
    with open("config/default.yaml") as f:
        config = yaml.safe_load(f)
    config['n_envs'] = 2          # two parallel mock envs
    config['total_timesteps'] = 10000
    config['seam_config'] = {'seam_type': 'line', 'start': [200,0,100], 'end': [600,0,100]}
    config['obstacles'] = []

    env = make_vec_env(lambda: WeldingEnv(config), n_envs=2)

    model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="./logs/",
                n_steps=512, batch_size=64, learning_rate=3e-4)
    model.learn(total_timesteps=10000)
    model.save("models/ppo_mock_test")
    print("Mini-training done. Check 'logs/' with TensorBoard.")

if __name__ == "__main__":
    main()
