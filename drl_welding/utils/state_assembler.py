import numpy as np

class StateAssembler:
    def __init__(self, config, seam):
        self.seam = seam
        self.state_dim = 41  # adjust based on components
        self.joint_limits = np.array(config["joint_limits"])  # rad

    def build(self, egm_wrapper):
        joints = egm_wrapper.get_joint_angles()
        pos, quat = egm_wrapper.get_tcp_pose()
        
        # Normalise joints
        norm_joints = joints / self.joint_limits
        
        # TCP features (already in base frame, normalise by workspace size)
        norm_pos = pos / 1000.0  # assume 1 m cube
        # Convert quaternion to 6D orientation (two axes of tool frame)
        rot_mat = self._quat_to_matrix(quat)
        tool_z = rot_mat[:,2]  # torch direction
        tool_x = rot_mat[:,0]  # travel direction
        orientation_6d = np.concatenate([tool_x, tool_z])
        
        # Seam related features
        closest_pt = self.seam.closest_point(pos)
        tangent = self.seam.tangent(closest_pt)
        look_ahead = self.seam.point_at_distance(closest_pt, 20.0)
        fraction = self.seam.fraction_completed(pos)
        # errors computed externally or from helpers
        standoff_error = ...  # computed from TCP Z distance to surface
        cross_error = ...     # lateral deviation
        speed = np.linalg.norm(...)  # estimated from previous pose
        
        # Obstacle distances (populated by collision module)
        obs_dists = self._get_obstacle_distances()
        
        state = np.concatenate([
            norm_joints,       # 6
            norm_pos,          # 3
            orientation_6d,    # 6
            norm_look_ahead,   # 3
            tangent,           # 3
            [fraction, standoff_error, cross_error, speed], # 4
            obs_dists,         # 3
            ...                # extra features up to 41
        ]).astype(np.float32)
        state = np.clip(state, -1, 1)
        return state

    def _quat_to_matrix(self, q):
        # standard conversion (omitted for brevity)
        pass
