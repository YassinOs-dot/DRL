from abb_robot_client import EGMSensorPoseClient
import numpy as np

class EGMWrapper:
    def __init__(self, host: str = "192.168.125.1", port: int = 6510):
        self.egm = EGMSensorPoseClient((host, port))
        # Initial read to synchronise
        state = self.egm.read()
        print(f"EGM connected. Joints: {state['robot']['joints']}")

    def get_joint_angles(self):
        msg = self.egm.read()
        return np.deg2rad(msg['robot']['joints'])

    def get_tcp_pose(self):
        msg = self.egm.read()
        cart = msg['cartesian']
        # pos in mm, quaternion (q1..q4)
        return np.array([cart['x'], cart['y'], cart['z']]), np.array(
            [cart['q1'], cart['q2'], cart['q3'], cart['q4']])

    def send_pose_correction(self, dx, dy, dz, dq):
        self.egm.send_pose_correction((dx, dy, dz, dq[0], dq[1], dq[2], dq[3]))
