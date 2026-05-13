"""Builds the normalised state vector for the RL agent."""
import numpy as np

class StateAssembler:
    def __init__(self, config, seam):
        self.config = config
        self.seam = seam
        self.joint_limits = np.array(config['joint_limits'])
        self.workspace_size = config.get('workspace_size', 1000)
        self.last_pos = None
        self.last_joints = None
        self.tcp_speed = 0.0
        self.joint_acc = np.zeros(6)
        # state dimension: joints(6) + pos(3) + tool_orientation_6d(6) + look_ahead(3)
        # + tangent(3) + progress(1) + standoff_err(1) + cross_err(1) + speed(1)
        # + obstacle_distances(3) = 28 dims
        self.state_dim = 28

    def reset(self, seam):
        self.seam = seam
        self.last_pos = None
        self.last_joints = None

    def build(self, egm_wrapper):
        joints = egm_wrapper.get_joint_angles()  # rad
        tcp_pos, tcp_quat = egm_wrapper.get_tcp_pose()

        # Normalise joints
        norm_joints = joints / self.joint_limits

        # Normalise position
        norm_pos = tcp_pos / self.workspace_size

        # Orientation: extract tool Z and X axes
        rot = self._quat_to_matrix(tcp_quat)
        tool_z = rot[:, 2]
        tool_x = rot[:, 0]

        # Seam features
        u = self.seam.closest_point(tcp_pos)
        closest_pt = self.seam.sample(u)
        tangent = self.seam.tangent(u)
        look_ahead = self.seam.point_at_distance(u, 20.0)
        fraction = self.seam.fraction_completed(tcp_pos)

        # Standoff error: distance from TCP to closest seam point along tool Z
        vec_to_seam = closest_pt - tcp_pos
        standoff = np.dot(vec_to_seam, tool_z)
        target_standoff = self.config['target_standoff']
        standoff_err = standoff - target_standoff

        # Cross-seam error: lateral distance perpendicular to tangent and tool Z
        perp = np.cross(tangent, tool_z)
        perp = perp / (np.linalg.norm(perp) + 1e-8)
        cross_err = np.dot(vec_to_seam, perp)

        # TCP speed (estimated from position change)
        if self.last_pos is not None:
            dt = self.config['policy_rate']
            self.tcp_speed = np.linalg.norm(tcp_pos - self.last_pos) / dt
        else:
            self.tcp_speed = 0.0

        # Joint accelerations
        if self.last_joints is not None:
            self.joint_acc = (joints - self.last_joints) / self.config['policy_rate']
        else:
            self.joint_acc = np.zeros(6)

        # Obstacle distances (mock: we can pass them in externally or compute here)
        # For now we fill with 1.0 (normalised max distance)
        obs_dists = np.array([1.0, 1.0, 1.0])

        # Concatenate
        state = np.concatenate([
            norm_joints,          # 6
            norm_pos,             # 3
            tool_x,               # 3
            tool_z,               # 3
            look_ahead / self.workspace_size,  # 3
            tangent,              # 3
            [fraction,
             np.clip(standoff_err / 50.0, -1, 1),
             np.clip(cross_err / 50.0, -1, 1),
             np.clip(self.tcp_speed / self.config['target_speed'] - 1.0, -1, 1)],
            obs_dists             # 3
        ]).astype(np.float32)

        # Store for next step
        self.last_pos = tcp_pos
        self.last_joints = joints

        return np.clip(state, -1, 1)

    def _quat_to_matrix(self, q):
        """Convert quaternion [w,x,y,z] to 3x3 rotation matrix."""
        w, x, y, z = q
        return np.array([
            [1-2*y*y-2*z*z, 2*x*y-2*z*w, 2*x*z+2*y*w],
            [2*x*y+2*z*w, 1-2*x*x-2*z*z, 2*y*z-2*x*w],
            [2*x*z-2*y*w, 2*y*z+2*x*w, 1-2*x*x-2*y*y]
        ])
