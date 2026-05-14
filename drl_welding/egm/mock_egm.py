"""Mock EGM client for local testing without RobotStudio."""
import numpy as np
import time

class MockEGMWrapper:
    def __init__(self, start_pos=np.array([200., 0., 100.])):
        self.joints = np.zeros(6)          # rad
        self.tcp_pos = start_pos.copy()
        self.tcp_quat = np.array([1., 0., 0., 0.])  # identity orientation
        self.collision_active = False

    def read_state(self):
        pass

    def get_joint_angles(self, rad=True):
        return self.joints.copy()

    def get_tcp_pose(self):
        return self.tcp_pos.copy(), self.tcp_quat.copy()

    def send_pose_correction(self, dx, dy, dz, dq):
        """Move TCP by the correction. dq = [qw, qx, qy, qz]."""
        self.tcp_pos += np.array([dx, dy, dz])
        # Simplified orientation update (accumulate quaternion)
        q = self.tcp_quat
        dq = np.array(dq)
        # Quaternion multiplication: q_new = dq * q (simplified)
        q_new = np.array([
            dq[0]*q[0] - dq[1]*q[1] - dq[2]*q[2] - dq[3]*q[3],
            dq[0]*q[1] + dq[1]*q[0] + dq[2]*q[3] - dq[3]*q[2],
            dq[0]*q[2] - dq[1]*q[3] + dq[2]*q[0] + dq[3]*q[1],
            dq[0]*q[3] + dq[1]*q[2] - dq[2]*q[1] + dq[3]*q[0]
        ])
        self.tcp_quat = q_new / np.linalg.norm(q_new)

    def set_collision(self, active):
        self.collision_active = active
