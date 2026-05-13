"""Unit tests for the environment without RobotStudio (using mock)."""
import numpy as np
from drl_welding.utils.seam_utils import SeamGenerator
from drl_welding.utils.collision import CollisionChecker
from drl_welding.utils.state_assembler import StateAssembler
import yaml

def test_seam():
    cfg = {'seam_type': 'line', 'start': [0,0,0], 'end': [100,0,0]}
    seam = SeamGenerator(cfg)
    p = seam.sample(0.5)
    assert np.allclose(p, [50,0,0])
    u = seam.closest_point([50,5,0])
    assert abs(u - 0.5) < 0.02
    print("Seam test passed.")

def test_collision():
    obstacles = [{'pos': [0,0,0], 'radius': 20}]
    checker = CollisionChecker(obstacles, robot_spheres=[np.array([0,0,30])])
    assert not checker.has_collision()
    checker.robot_spheres = [np.array([0,0,10])]
    assert checker.has_collision()
    print("Collision test passed.")

if __name__ == "__main__":
    test_seam()
    test_collision()
    print("All tests passed.")
