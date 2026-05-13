"""Gymnasium environment for DRL welding."""
import time
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from drl_welding.egm.egm_client import EGMWrapper
from drl_welding.utils.seam_utils import SeamGenerator
from drl_welding.utils.collision import CollisionChecker
from drl_welding.utils.state_assembler import StateAssembler
from drl_welding.reward.reward_calculator import RewardCalculator

class WeldingEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.dt = config['policy_rate']
        self.max_steps = config.get('max_episode_steps', 1000)
        self.current_step = 0

        # EGM communication
        self.egm = EGMWrapper(config['egm_host'], config['egm_port'])

        # Seam
        self.seam = SeamGenerator(config.get('seam_config', {'seam_type':'random'}))

        # Collision checker with obstacles from config
        obstacles = config.get('obstacles', [])
        self.collision = CollisionChecker(obstacles)

        # State assembler
        self.state_assembler = StateAssembler(config, self.seam)

        # Reward calculator
        self.reward_calc = RewardCalculator(config['reward_weights'],
                                           config['target_speed'],
                                           config['target_standoff'])

        # Action space: [vx, vy, vz, wx, wy, wz] normalised to [-1,1]
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(6,), dtype=np.float32)

        # Observation space
        obs_dim = self.state_assembler.state_dim
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(obs_dim,), dtype=np.float32)

        # State variables for reward
        self.last_pos = None
        self.last_closest_u = None

    def step(self, action):
        # Scale action
        v_lin = action[:3] * self.config['max_linear_vel']
        v_ang = action[3:] * np.deg2rad(self.config['max_angular_vel'])

        # Integrate to pose correction
        dx, dy, dz = v_lin * self.dt
        dq = self._rotvec_to_quat(v_ang * self.dt)
        self.egm.send_pose_correction(dx, dy, dz, dq)

        # Wait for policy period by reading EGM messages
        t_end = time.time() + self.dt
        while time.time() < t_end:
            self.egm.read_state()

        # Build state
        state = self.state_assembler.build(self.egm)

        # Compute reward
        tcp_pos, tcp_quat = self.egm.get_tcp_pose()
        state_dict = self._build_reward_dict(tcp_pos, tcp_quat)
        reward, done, info = self.reward_calc.compute(state_dict, self.seam, self.collision, self.current_step)

        self.current_step += 1
        truncated = self.current_step >= self.max_steps
        return state, reward, done or truncated, truncated, info

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        # Re-randomise seam and obstacles
        self.seam = SeamGenerator(self.config.get('seam_config', {'seam_type':'random'}))
        obstacles = self._generate_obstacles()
        self.collision.set_obstacles(obstacles)
        self.state_assembler.reset(self.seam)
        # Re-sync EGM (RAPID program must be running)
        time.sleep(0.1)
        self.egm.read_state()
        state = self.state_assembler.build(self.egm)
        self.last_pos = self.egm.get_tcp_pose()[0]
        u = self.seam.closest_point(self.last_pos)
        self.last_closest_u = u
        return state, {}

    def _build_reward_dict(self, tcp_pos, tcp_quat):
        u = self.seam.closest_point(tcp_pos)
        tangent = self.seam.tangent(u)
        # Progress along seam
        if self.last_closest_u is not None:
            du = max(0.0, u - self.last_closest_u)
            progress_mm = du * self.seam.total_length
        else:
            progress_mm = 0.0
        self.last_closest_u = u

        # Orientation error (angle between tool Z and surface normal = tangent cross product? simplified)
        # For simplicity we use the angle between tool Z and negative surface normal (Z perpendicular to seam)
        tool_z = self._quat_to_matrix(tcp_quat)[:, 2]
        # Desired orientation: Z perpendicular to surface, X along tangent.
        # Here we approximate desired tool Z as pointing toward the workpiece (e.g., [0,0,-1] in world)
        desired_z = np.array([0, 0, -1])  # assumes horizontal welding table
        angle_err = np.arccos(np.clip(np.dot(tool_z, desired_z), -1, 1))

        # Standoff error: use state assembler's computation
        standoff_err = self.state_assembler._standoff_error if hasattr(self.state_assembler, '_standoff_error') else 0.0
        cross_err = self.state_assembler._cross_error if hasattr(self.state_assembler, '_cross_error') else 0.0

        return {
            'progress_mm': progress_mm,
            'torch_orientation_error': angle_err,
            'joint_acc': self.state_assembler.joint_acc,
            'tcp_speed': self.state_assembler.tcp_speed,
            'standoff_error': standoff_err,
            'cross_seam_error': cross_err,
            'seam_fraction': self.seam.fraction_completed(tcp_pos),
        }

    def _generate_obstacles(self):
        cfg = self.config
        n_obs = np.random.randint(*cfg.get('obstacle_count_range', [1, 3]))
        obs = []
        for _ in range(n_obs):
            pos = np.random.uniform(low=[100, -200, 0], high=[600, 200, 300], size=3)
            radius = np.random.uniform(*cfg.get('obstacle_radius_range', [30, 80]))
            obs.append({'pos': pos, 'radius': radius})
        return obs

    def _rotvec_to_quat(self, rotvec):
        angle = np.linalg.norm(rotvec)
        if angle < 1e-6:
            return np.array([1, 0, 0, 0])
        axis = rotvec / angle
        return np.concatenate([[np.cos(angle/2)], np.sin(angle/2) * axis])

    def _quat_to_matrix(self, q):
        w, x, y, z = q
        return np.array([
            [1-2*y*y-2*z*z, 2*x*y-2*z*w, 2*x*z+2*y*w],
            [2*x*y+2*z*w, 1-2*x*x-2*z*z, 2*y*z-2*x*w],
            [2*x*z-2*y*w, 2*y*z+2*x*w, 1-2*x*x-2*y*y]
        ])
