"""Collision checking using simple geometric models (spheres)."""
import numpy as np

class CollisionChecker:
    def __init__(self, obstacles=None, robot_segments=None):
        """
        obstacles: list of dicts with 'pos' (3,) and 'radius' (float)
        robot_segments: list of points along the robot body (simplified).
        """
        self.obstacles = obstacles if obstacles else []
        # Approximate robot as a few spheres along the arm (in base frame)
        self.robot_spheres = robot_segments if robot_segments else [
            np.array([0,0,300]),   # shoulder
            np.array([0,0,600]),   # elbow
            np.array([0,0,800]),   # wrist
        ]
        self._min_dist = 999.0

    def set_obstacles(self, obstacles):
        self.obstacles = obstacles

    def update_robot_pose(self, tcp_pose):
        """
        Given TCP pose, approximate positions of robot spheres.
        For simplicity we just shift the predefined spheres with the TCP's base offset.
        In a real scenario, forward kinematics would be used.
        """
        # Here we assume the base frame is fixed and the robot segments are static;
        # for a more accurate check, compute FK from joint angles.
        pass

    def min_distance(self):
        """Return minimum distance between any robot sphere and obstacle."""
        min_d = 1e9
        for obs in self.obstacles:
            oc = np.array(obs['pos'])
            r = obs['radius']
            for sp in self.robot_spheres:
                d = np.linalg.norm(sp - oc) - r
                if d < min_d:
                    min_d = d
        self._min_dist = min_d
        return max(0.0, min_d)

    def has_collision(self):
        """True if any robot sphere intersects an obstacle."""
        return self.min_distance() < 1.0  # tolerance
