from math_utility import joint_angle
import numpy as np
class SlidingWindow():

    WINDOW_SIZE = 18  # 10–15 frames (~0.33–0.5 sec at 30 FPS)
    STEP_SIZE = 9 #frames
    INTEREST_PARTS = ["LElbow", "RShoulder", "LShoulder", "RElbow", "RWrist", "LWrist" ]
    ANGLE_DEFINITIONS = {
                        "L_elbow": ("LShoulder", "LElbow", "LWrist"),
                        "R_elbow": ("RShoulder", "RElbow", "RWrist"),

                        "L_shoulder": ("MidHip", "LShoulder", "LElbow"),
                        "R_shoulder": ("MidHip", "RShoulder", "RElbow"),
                    }

    def __init__(self, sliding_window_id, start_frame, end_frame, person):

        self.id = sliding_window_id
        self.start_frame = start_frame
        self.end_frame = end_frame
        self.person = person
    
    def build_features(self):
        """
        Build all features for this sliding window.
        """
        self.motion_energy = self.compute_motion_energy()
        self.mean_velocity = self.compute_mean_velocity()
        self.motion_persistence = self.compute_motion_persistence()
        self.velocity_variance = self.compute_velocity_variance()
        self.distal_proximal_motion_ratio = self.compute_distal_proximal_motion_ratio()
        self.directional_consistency = self.compute_directional_consistency()
        self.lh_energy, self.rh_energy = self.compute_motion_energy_hands()
        computed_angles = self.compute_joint_angles_and_velocity()
        self.max_angle = max(computed_angles['L_elbow_angle_mean'],computed_angles['R_elbow_angle_mean'],computed_angles['L_shoulder_angle_mean'],computed_angles['R_shoulder_angle_mean'],)/self.WINDOW_SIZE
        self.max_angular_velocity = max(computed_angles['L_elbow_angvel_mean'],computed_angles['R_elbow_angvel_mean'],computed_angles['L_shoulder_angvel_mean'],computed_angles['R_shoulder_angvel_mean'],)/self.WINDOW_SIZE
        self.max_acceleration = self.compute_max_acceleration()

    def compute_joint_angles_and_velocity(self, fps=24):
            """
            Compute mean joint angles and mean angular velocities over a window.

            Returns:
                features: dict with keys:
                    "{joint}_angle_mean": mean angle in radians
                    "{joint}_angvel_mean": mean angular velocity in radians/sec
            """
            angles = {name: [] for name in SlidingWindow.ANGLE_DEFINITIONS}

            # 1. Compute per-frame angles
            for frame_idx in range(self.start_frame + 1, self.end_frame + 1):
                for name, (p1, p2, p3) in SlidingWindow.ANGLE_DEFINITIONS.items():
                    body_part1 = self.person.get_body_part(p1)
                    body_part2 = self.person.get_body_part(p2)
                    body_part3 = self.person.get_body_part(p3)

                    try:
                        # Get normalized coordinates
                        p1_xy = body_part1.get_normalized_coordinates(frame_idx)
                        p2_xy = body_part2.get_normalized_coordinates(frame_idx)
                        p3_xy = body_part3.get_normalized_coordinates(frame_idx)

                        angle = joint_angle(p1_xy, p2_xy, p3_xy)
                    except Exception:
                        angle = np.nan  # Missing joints

                    angles[name].append(angle)

            features = {}

            # 2. Convert to arrays, compute mean angle and mean angular velocity
            for name, angle_list in angles.items():
                angle_array = np.array(angle_list)

                # Mean angle over the window
                mean_angle = np.nanmean(angle_array)
                features[f"{name}_angle_mean"] = mean_angle

                # Angular velocity (radians/sec)
                if len(angle_array) > 1:
                    ang_vel = np.gradient(angle_array) * fps  # d(angle)/dt
                    mean_ang_vel = np.nanmean(np.abs(ang_vel))
                else:
                    mean_ang_vel = np.nan

                features[f"{name}_angvel_mean"] = mean_ang_vel

            return features


  
    def compute_motion_energy(self):
        energy_per_part = []

        for part_name, body_part in self.person.body.items():
            if part_name not in SlidingWindow.INTEREST_PARTS:
                continue

            part_energy = 0.0
            for frame_idx in range(self.start_frame + 1, self.end_frame + 1):
                if body_part.confident(frame_idx):
                    v = body_part.get_velocity_magnitude(frame_idx)
                    part_energy += v ** 2

            energy_per_part.append(part_energy)

        # Take max across parts (dominant motion)
        energy = max(energy_per_part) if energy_per_part else 0.0

        # Normalize by window size
        energy /= (self.WINDOW_SIZE)
        return energy

    def compute_motion_energy_hands(self):
        lh_energy = 0.0
        rh_energy = 0.0

        for part_name, body_part in self.person.left_hand.items():
            for frame_idx in range(self.start_frame + 1, self.end_frame + 1):
                if body_part.confident(frame_idx):
                    v = body_part.get_velocity_magnitude(frame_idx)
                    lh_energy += v ** 2

        for part_name, body_part in self.person.right_hand.items():
            for frame_idx in range(self.start_frame + 1, self.end_frame + 1):
                if body_part.confident(frame_idx):
                    v = body_part.get_velocity_magnitude(frame_idx)
                    rh_energy += v ** 2

        # Normalize by window size
        lh_energy /= (self.WINDOW_SIZE)
        rh_energy /= (self.WINDOW_SIZE)
        return (lh_energy, rh_energy)
    def compute_mean_velocity(self):
        total_velocity = 0.0
        count = 0

        for part_name, body_part in self.person.body.items():
            if part_name not in SlidingWindow.INTEREST_PARTS:
                continue
            for frame_idx in range(self.start_frame + 1, self.end_frame + 1):
                velocity = body_part.get_velocity_magnitude(frame_idx)
                total_velocity += velocity
                count += 1

        return total_velocity / count if count > 0 else 0.0
    def compute_max_acceleration(self):
        accelerations = []

        for part_name, body_part in self.person.body.items():
            if part_name not in SlidingWindow.INTEREST_PARTS:
                continue
            for frame_idx in range(self.start_frame + 1, self.end_frame + 1):
                if body_part.confident(frame_idx):
                    acc = body_part.get_acceleration_magnitude(frame_idx)
                    accelerations.append(acc)

        return max(accelerations)
    def compute_motion_persistence(self, threshold=0.1):
        total_frames = self.WINDOW_SIZE
        persistence_per_part = []

        for part_name, body_part in self.person.body.items():
            if part_name not in SlidingWindow.INTEREST_PARTS:
                continue
            moving_frames = 0
            for frame_idx in range(self.start_frame + 1, self.end_frame + 1):
                v = body_part.get_velocity_magnitude(frame_idx)
                if v > threshold:
                    moving_frames += 1
            persistence_per_part.append(moving_frames / total_frames)

        # Average persistence across parts
        return sum(persistence_per_part) / len(persistence_per_part) if persistence_per_part else 0.0
    
    def compute_velocity_variance(self):
        velocities = []

        for part_name, body_part in self.person.body.items():
            if part_name not in SlidingWindow.INTEREST_PARTS:
                continue
            for frame_idx in range(self.start_frame + 1, self.end_frame + 1):
                velocity = body_part.get_velocity_magnitude(frame_idx)
                velocities.append(velocity)

        if len(velocities) == 0:
            return 0.0

        mean_velocity = sum(velocities) / len(velocities)
        variance = sum((v - mean_velocity) ** 2 for v in velocities) / len(velocities)

        return variance
    
    def compute_distal_proximal_motion_ratio(self, eps=1e-5):
        distal_parts = ["RWrist", "LWrist", "RElbow", "LElbow"]
        proximal_parts = ["RShoulder", "LShoulder"]

        distal_motion = 0.0
        proximal_motion = 0.0

        for part_name, body_part in self.person.body.items():
            if part_name in distal_parts:
                for frame_idx in range(self.start_frame + 1, self.end_frame + 1):
                    distal_motion += body_part.get_velocity_magnitude(frame_idx)
            elif part_name in proximal_parts:
                for frame_idx in range(self.start_frame + 1, self.end_frame + 1):
                    proximal_motion += body_part.get_velocity_magnitude(frame_idx)

        return distal_motion / (proximal_motion + eps)


    def compute_directional_consistency(self, min_velocity_threshold=0.1):
        parts = ["RWrist", "LWrist", "RElbow", "LElbow"]
        vx_sum, vy_sum = 0.0, 0.0
        magnitude_sum = 0.0

        for part_name in parts:
            body_part = self.person.body.get(part_name)
            if not body_part:
                continue

            for frame_idx in range(self.start_frame + 1, self.end_frame + 1):
                vx, vy = body_part.get_velocity_vector(frame_idx)

                vel_mag = (vx**2 + vy**2)**0.5
                if vel_mag < min_velocity_threshold:
                    continue  # ignore tiny motions

                vx_sum += vx
                vy_sum += vy
                magnitude_sum += vel_mag

        if magnitude_sum == 0:
            return 0.0  # no motion

        # Ratio of net motion to total motion
        consistency = (vx_sum**2 + vy_sum**2)**0.5 / magnitude_sum
        return min(max(consistency, 0.0), 1.0)  # clamp to [0,1]
   
    def compute_score(self,motion_energy_weight,
                         mean_velocity_weight,
                         distal_proximal_weight,
                         persistence_weight,
                         directional_weight,
                         hands_weight,
                         acc_weight,
                         max_angular_velocity_weight):
        # Normalize features for scoring (simple min-max idea)
        # You can tweak min/max empirically after plotting distributions
        # Here we assume features are already roughly in [0,1] range after normalization
        self.score = (
            motion_energy_weight * self.motion_energy +
            mean_velocity_weight * self.mean_velocity +
            # distal_proximal_weight * self.distal_proximal_motion_ratio +
            persistence_weight * self.motion_persistence +
            max_angular_velocity_weight * self.max_angular_velocity +
            directional_weight * self.directional_consistency+
            hands_weight * max(self.lh_energy, self.rh_energy) +
            acc_weight * self.max_acceleration
        )
    def contains_gesture(self,
                         motion_energy_weight=0.65,
                         mean_velocity_weight=0.3,
                         distal_proximal_weight=0.05,
                         persistence_weight=0.5,
                         directional_weight=0.5,
                         max_angular_velocity_weight=0.9,
                         hands_weight=0.45,
                         acc_weight=0.2,
                         score_threshold=0.2):
        self.compute_score(motion_energy_weight, mean_velocity_weight, distal_proximal_weight, persistence_weight, directional_weight,hands_weight, acc_weight, max_angular_velocity_weight)
        if self.max_acceleration < 0.05 and self.max_angular_velocity < 0.028: 
            return False
        return self.score >= score_threshold
    
    def should_merge_with(self, other_window, 
                         max_temporal_gap=6,
                         feature_similarity_threshold=0.7):
        """
        Determine if this window should be merged with another window.
        
        Two windows are merged if:
        1. They are temporally close (within max_temporal_gap frames)
        2. Their motion features are similar (above similarity threshold)
        
        Args:
            other_window (SlidingWindow): The other window to compare with.
            max_temporal_gap (int): Maximum frames allowed between windows to merge.
            feature_similarity_threshold (float): Minimum similarity (0-1) of normalized features.
        
        Returns:
            bool: True if windows should be merged, False otherwise.
        """
        # Ensure both windows have features computed
        if not hasattr(self, 'motion_energy'):
            self.build_features()
        if not hasattr(other_window, 'motion_energy'):
            other_window.build_features()
        
        # Check temporal proximity
        temporal_gap = min(
            abs(other_window.start_frame - self.end_frame),
            abs(self.start_frame - other_window.end_frame)
        )
        
        if temporal_gap > max_temporal_gap:
            return False
        return True