#!/usr/bin/env python
"""Train PPO agent on the welding environment."""
import yaml
import argparse
import os
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
from drl_welding.envs.welding_env import WeldingEnv

def make_env(config, rank):
    def _init():
        env = WeldingEnv(config)
        env = Monitor(env)
        return env
    return _init

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config/default.yaml', help='Path to config file')
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    n_envs = config.get('n_envs', 4)
    env = SubprocVecEnv([make_env(config, i) for i in range(n_envs)])
    # Normalise observations and rewards (optional)
    # env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.)

    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        tensorboard_log="./logs/",
        n_steps=config['n_steps'],
        batch_size=config['batch_size'],
        learning_rate=config['learning_rate'],
        gamma=config['gamma'],
        gae_lambda=config['gae_lambda'],
        ent_coef=config['ent_coef'],
        policy_kwargs={'net_arch': config['net_arch']}
    )

    model.learn(total_timesteps=config['total_timesteps'])
    os.makedirs('models', exist_ok=True)
    model.save("models/ppo_welder")
    print("Training finished, model saved.")

if __name__ == "__main__":
    main()
