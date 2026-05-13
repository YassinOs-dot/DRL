import numpy as np

class RewardCalculator:
    def __init__(self, weights, target_speed=10.0, target_standoff=12.0):
        self.w = weights
        self.target_speed = target_speed
        self.target_standoff = target_standoff

    def compute(self, state_dict, seam, collision_checker, step):
        """
        state_dict should contain:
          - tcp_pos
          - tcp_quat
          - joint_acc
          - tcp_speed
          - standoff_error
          - cross_seam_error
          - torch_orientation_error (radians, from state assembler's tool Z deviation)
          - seam_fraction
        """
        r = 0.0
        done = False
        info = {}

        # Progress: displacement along seam tangent since last step
        progress = state_dict.get('progress_mm', 0.0)
        r += self.w['progress'] * progress

        # Torch orientation error
        angle_err = state_dict.get('torch_orientation_error', 0.0)
        r -= self.w['alignment'] * angle_err

        # Smoothness (joint accelerations)
        joint_acc = state_dict.get('joint_acc', np.zeros(6))
        r -= self.w['smooth'] * np.sum(joint_acc**2)

        # Speed penalty if outside ±10%
        speed = state_dict.get('tcp_speed', self.target_speed)
        rel_err = abs(speed - self.target_speed) / self.target_speed
        if rel_err > 0.1:
            r -= self.w['speed_penalty'] * (rel_err - 0.1)

        # Standoff error
        standoff_err = state_dict.get('standoff_error', 0.0)
        r -= self.w['standoff'] * abs(standoff_err)

        # Cross-seam error
        cross_err = state_dict.get('cross_seam_error', 0.0)
        r -= self.w['standoff'] * abs(cross_err)  # reuse standoff weight or separate

        # Collision / proximity
        min_dist = collision_checker.min_distance()
        if collision_checker.has_collision():
            r += self.w['collision_penalty']
            done = True
            info['collision'] = True
        elif min_dist < 20:
            r -= self.w['close_penalty'] * (20 - min_dist) / 20.0

        # Completion
        fraction = state_dict.get('seam_fraction', 0.0)
        if fraction >= 1.0:
            r += self.w['completion_bonus']
            done = True
            info['success'] = True

        return r, done, info
