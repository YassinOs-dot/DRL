"""Parametric weld seam definitions and helpers."""
import numpy as np
from scipy.interpolate import CubicSpline

class SeamGenerator:
    def __init__(self, seam_config):
        self.config = seam_config
        self.points = None        # Nx3 waypoints
        self.spline = None        # CubicSpline per axis
        self.total_length = 0.0
        self.param_range = [0.0, 1.0]
        self._generate()

    def _generate(self):
        cfg = self.config
        stype = cfg.get('seam_type', 'random')
        if stype == 'random':
            self._random_seam()
        elif stype == 'line':
            start = np.array(cfg['start'])
            end = np.array(cfg['end'])
            self._from_points(np.array([start, end]))
        elif stype == 'spline':
            ctrl = np.array(cfg['control_points'])
            self._from_points(ctrl)
        elif stype == 'circular_arc':
            self._circular_arc(cfg)
        else:
            raise ValueError(f"Unknown seam type: {stype}")

    def _random_seam(self):
        length = np.random.uniform(*self.config.get('seam_length_range', [100, 400]))
        curv = self.config.get('curvature_max', 0.02)
        # Create a random Bezier-like spline with 4 control points
        start = np.array([200, 0, 100])
        end = start + np.array([length, 0, 0])
        c1 = start + np.random.uniform(-50, 50, 3)
        c2 = end + np.random.uniform(-50, 50, 3)
        self._from_points(np.array([start, c1, c2, end]))

    def _from_points(self, points):
        self.points = points
        n = len(points)
        t = np.linspace(0, 1, n)
        # Fit a cubic spline for x, y, z separately
        self.spline = CubicSpline(t, points, axis=0, bc_type='natural')
        self.param_range = [0.0, 1.0]
        self.total_length = self._compute_length()

    def _circular_arc(self, cfg):
        center = np.array(cfg['center'])
        radius = cfg['radius']
        start_angle = np.deg2rad(cfg['start_angle'])
        end_angle = np.deg2rad(cfg['end_angle'])
        angles = np.linspace(start_angle, end_angle, 50)
        points = center + radius * np.column_stack([np.cos(angles), np.sin(angles), np.zeros(50)])
        self._from_points(points)

    def _compute_length(self, n_samples=200):
        t = np.linspace(0, 1, n_samples)
        pts = self.spline(t)
        diffs = np.diff(pts, axis=0)
        return np.sum(np.linalg.norm(diffs, axis=1))

    def sample(self, u):
        """Return 3D point at parameter u in [0,1]."""
        return self.spline(u)

    def tangent(self, u):
        """Return normalised tangent at u."""
        du = 1e-4
        p0 = self.spline(u - du if u > du else 0)
        p1 = self.spline(u + du if u < 1-du else 1)
        t = p1 - p0
        norm = np.linalg.norm(t)
        return t / norm if norm > 1e-8 else np.array([1, 0, 0])

    def closest_point(self, pos):
        """Find parameter u for the closest point on the spline to 'pos'."""
        # Coarse search then refine
        us = np.linspace(0, 1, 200)
        pts = self.spline(us)
        dists = np.linalg.norm(pts - pos, axis=1)
        u_init = us[np.argmin(dists)]
        # Fine Newton search (simple gradient descent)
        for _ in range(5):
            p = self.spline(u_init)
            dp = self.spline.derivative()(u_init)
            err = p - pos
            grad = 2 * np.dot(dp, err)
            u_init -= 0.01 * grad
            u_init = np.clip(u_init, 0.0, 1.0)
        return u_init

    def point_at_distance(self, u, dist):
        """Return point 'dist' mm ahead of u along the seam."""
        t = self.tangent(u)
        return self.sample(u) + dist * t

    def fraction_completed(self, pos):
        """Estimated fraction of seam completed (0..1) based on TCP projection."""
        u = self.closest_point(pos)
        # Integrate length from 0 to u
        t_samples = np.linspace(0, u, 30)
        pts = self.spline(t_samples)
        segs = np.diff(pts, axis=0)
        arc_length = np.sum(np.linalg.norm(segs, axis=1))
        return min(1.0, arc_length / self.total_length) if self.total_length > 0 else 0.0
