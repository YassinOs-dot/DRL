import gymnasium as gym
import numpy as np
from drl_welding.egm.egm_client import EGMWrapper
from drl_welding.reward.reward_calculator import RewardCalculator
from drl_welding.utils.state_assembler import StateAssembler
from drl_welding.utils.seam_utils import SeamGenerator
from drl_welding.utils.collision import CollisionChecker

class WeldingEnv(gym.Env):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.egm = EGMWrapper(config["egm_host"], config["egm_port"])
        self.seam = SeamGenerator(config["seam_type"])  # randomisable
        self.reward_calc = RewardCalculator(config["reward_weights"])
        self.state_asm = StateAssembler(config, self.seam)
        self.collision = CollisionChecker()
        
        # Action space: 6D velocity (vx,vy,vz,ωx,ωy,ωz) in tool frame, scaled to [-1,1]
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(6,), dtype=np.float32)
        # State space size defined by StateAssembler
        self.observation_space = gym.spaces.Box(
            low=-1, high=1, shape=(self.state_asm.state_dim,), dtype=np.float32
        )
        self.dt = config["policy_rate"]  # e.g. 0.016 s
        self.max_episode_steps = config.get("max_steps", 1000)
        self.current_step = 0

    def step(self, action):
        # Scale action to real velocities
        v_lin = action[:3] * self.config["max_linear_vel"]  # mm/s
        v_ang = action[3:] * self.config["max_angular_vel"] # rad/s
        
        # Integrate to get pose correction
        dx, dy, dz = v_lin * self.dt
        dq = self._rotvec_to_quat(v_ang * self.dt)
        self.egm.send_pose_correction(dx, dy, dz, dq)
        
        # Wait for the full dt by reading multiple EGM messages
        for _ in range(int(self.dt / 0.004)):
            self.egm.egm.read()
        
        # Build state
        state = self.state_asm.build(self.egm)
        
        # Compute reward and check termination
        reward, done, info = self.reward_calc.compute(
            state, self.seam, self.collision, self.current_step
        )
        self.current_step += 1
        truncated = self.current_step >= self.max_episode_steps
        return state, reward, done or truncated, truncated, info

    def reset(self, seed=None, options=None):
        self._restart_simulation()          # via RobotStudio .NET API or digital signal
        self.seam.randomize()               # new geometry for domain randomisation
        self.state_asm.reset(self.seam)
        self.current_step = 0
        return self.state_asm.build(self.egm), {}

    def _restart_simulation(self):
        # Implementation using pythonnet to call RobotStudio API, or a simple
        # digital output toggle that restarts the RAPID program.
        pass

    def _rotvec_to_quat(self, rotvec):
        angle = np.linalg.norm(rotvec)
        if angle < 1e-6:
            return np.array([1,0,0,0])
        axis = rotvec / angle
        return np.concatenate([[np.cos(angle/2)], np.sin(angle/2) * axis])
