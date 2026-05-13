import yaml
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from drl_welding.envs.welding_env import WeldingEnv

def make_env(config, rank):
    def _init():
        env = WeldingEnv(config)
        return env
    return _init

if __name__ == "__main__":
    with open("config/default.yaml") as f:
        config = yaml.safe_load(f)
    
    num_envs = 4
    env = SubprocVecEnv([make_env(config, i) for i in range(num_envs)])
    
    model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="./logs/",
                n_steps=2048, batch_size=64, learning_rate=3e-4,
                gamma=0.99, gae_lambda=0.95, ent_coef=0.01,
                policy_kwargs=dict(net_arch=[256,256], activation_fn='tanh'))
    
    model.learn(total_timesteps=2_000_000)
    model.save("ppo_welder")
