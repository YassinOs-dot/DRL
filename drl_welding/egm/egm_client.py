"""EGM communication wrapper using abb_robot_client."""
import numpy as np
from abb_robot_client import EGMSensorPoseClient

class EGMWrapper:
    def __init__(self, host="192.168.125.1", port=6510):
        self.egm = EGMSensorPoseClient((host, port))
        # First read to synchronise
        state = self.egm.read()
        self._last_msg = state
        self._joint_vel = np.zeros(6)
        print(f"EGM connected. Initial joints: {state['robot']['joints']}")

    def read_state(self):
        """Update and return the latest EGM message."""
        self._last_msg = self.egm.read()
        return self._last_msg

    def get_joint_angles(self, rad=True):
        msg = self.read_state()
        joints = np.array(msg['robot']['joints'])
        return np.deg2rad(joints) if rad else joints

    def get_tcp_pose(self):
        msg = self.read_state()
        cart = msg['cartesian']
        pos = np.array([cart['x'], cart['y'], cart['z']], dtype=np.float64)
        quat = np.array([cart['q1'], cart['q2'], cart['q3'], cart['q4']], dtype=np.float64)
        return pos, quat

    def send_pose_correction(self, dx, dy, dz, dq):
        """
        Send a relative pose correction (sensor correction frame).
        dq = [qw, qx, qy, qz]
        """
        self.egm.send_pose_correction((dx, dy, dz, dq[0], dq[1], dq[2], dq[3]))
