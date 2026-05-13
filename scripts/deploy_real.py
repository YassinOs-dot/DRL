#!/usr/bin/env python
"""Deploy trained policy on a real ABB robot via EGM."""
import argparse
from stable_baselines3 import PPO
from drl_welding.envs.welding_env import WeldingEnv

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='models/ppo_welder', help='Path to model')
    parser.add_argument('--host', default='192.168.125.1', help='Robot controller IP')
    args = parser.parse_args()

    model = PPO.load(args.model)

    config = {
        'egm_host': args.host,
        'egm_port': 6510,
        'policy_rate': 0.016,
        'max_linear_vel': 30,      # slower for safety
        'max_angular_vel': 15,
        'target_speed': 8.0,
        'target_standoff': 12.0,
        'joint_limits': [2.967, 1.745, 2.967, 3.142, 2.094, 6.283],
        'reward_weights': {},
        'seam_config': {'seam_type': 'line', 'start': [200,0,100], 'end': [400,0,100]},
        'obstacles': [],
        'max_episode_steps': 2000
    }
    env = WeldingEnv(config)
    obs, _ = env.reset()
    print("Starting real robot execution. Press Ctrl+C to stop.")
    try:
        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
            if done or truncated:
                print("Episode finished.")
                break
    except KeyboardInterrupt:
        print("Stopped by user.")

if __name__ == "__main__":
    main()
