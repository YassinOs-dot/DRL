"""Simple connectivity test for EGM (requires RobotStudio running)."""
import time
from drl_welding.egm.egm_client import EGMWrapper

def test_connection():
    egm = EGMWrapper(host="192.168.125.1", port=6510)
    print("Reading initial state...")
    joints = egm.get_joint_angles()
    print(f"Joints (rad): {joints}")
    print("Sending zero correction...")
    for _ in range(10):
        egm.send_pose_correction(0, 0, 0, (1, 0, 0, 0))
        time.sleep(0.004)
    print("Test passed.")

if __name__ == "__main__":
    test_connection()
