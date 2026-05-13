import numpy as np

class RewardCalculator:
    def __init__(self, weights):
        self.w = weights

    def compute(self, state_dict, seam, collision_checker, step):
        r = 0.0
        done = False
        info = {}
        
        # Progress along seam (dot product of movement with tangent)
        progress = max(0, state_dict['tcp_displacement_along_seam'])
        r += self.w['progress'] * progress
        
        # Torch orientation error
        angle_err = state_dict['torch_orientation_error']  # rad
        r -= self.w['alignment'] * angle_err
        
        # Smoothness (joint accelerations)
        joint_acc = state_dict.get('joint_acc', np.zeros(6))
        r -= self.w['smooth'] * np.sum(joint_acc**2)
        
        # Speed deviation
        speed_err = abs(state_dict['tcp_speed'] - self.w['target_speed'])
        r -= self.w['speed_penalty'] * max(0, speed_err/self.w['target_speed'] - 0.1)
        
        # Standoff
        standoff_err = abs(state_dict['standoff'] - self.w['target_standoff'])
        r -= self.w['standoff'] * standoff_err
        
        # Collision / proximity
        if collision_checker.has_collision():
            r += self.w['collision_penalty']
            done = True
            info['collision'] = True
        else:
            min_dist = collision_checker.min_distance()
            if min_dist < 20:
                r -= self.w['close_penalty'] * (20 - min_dist) / 20.0
        
        # Seam completion
        if state_dict['seam_fraction'] >= 1.0:
            r += self.w['completion_bonus']
            done = True
            info['success'] = True
        
        return r, done, info
