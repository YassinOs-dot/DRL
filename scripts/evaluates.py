#!/usr/bin/env python
"""Evaluate trained policy on test scenes."""
import yaml
import argparse
import numpy as np
from stable_baselines3 import PPO
from drl_welding.envs.welding_env import WeldingEnv
from drl_welding.utils.seam_utils import SeamGenerator

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='models/ppo_welder', help='Path to trained model')
    parser.add_argument('--scenes', default='config/test_scenes.yaml', help='Test scenes config')
    parser.add_argument('--render', action='store_true')
    args = parser.parse_args()

    model = PPO.load(args.model)

    with open(args.scenes) as f:
        scenes = yaml.safe_load(f)['scenes']

    results = []
    for scene in scenes:
        print(f"Evaluating scene: {scene['name']}")
        # Build environment with this specific scene (non-random)
        config = {
            'egm_host': '192.168.125.1',  # change as needed
            'egm_port': 6510,
            'policy_rate': 0.016,
            'max_linear_vel': 50,
            'max_angular_vel': 30,
            'target_speed': 10.0,
            'target_standoff': 12.0,
            'joint_limits': [2.967, 1.745, 2.967, 3.142, 2.094, 6.283],
            'reward_weights': {  # dummy, not used in eval
                'progress':20, 'alignment':10, 'smooth':0.001, 'speed_penalty':5,
                'standoff':2, 'close_penalty':5, 'collision_penalty':-100, 'completion_bonus':200
            },
            'seam_config': scene,
            'obstacles': scene.get('obstacles', []),
            'max_episode_steps': 2000
        }
        env = WeldingEnv(config)
        obs, _ = env.reset()
        total_reward = 0
        collisions = 0
        success = False
        path_length = 0.0
        last_pos = None

        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
            total_reward += reward
            if info.get('collision'):
                collisions += 1
            if info.get('success'):
                success = True
            # Estimate path length
            cur_pos = env.egm.get_tcp_pose()[0]
            if last_pos is not None:
                path_length += np.linalg.norm(cur_pos - last_pos)
            last_pos = cur_pos
            if done or truncated:
                break

        results.append({
            'scene': scene['name'],
            'success': success,
            'collisions': collisions,
            'path_length_mm': path_length,
            'total_reward': total_reward
        })
        print(f"  Success: {success}, Collisions: {collisions}, Path: {path_length:.1f} mm")

    print("\nSummary:")
    for r in results:
        print(r)

if __name__ == "__main__":
    main()
