from gesture_manager.features import Features
from config import COCO_PARTS, HAND_PARTS, CORRECTION_FACTOR_HANDS, CORRECTION_FACTOR_VAR
import numpy as np
from math_utility import joint_angle
import math


class FeaturesManager:
    INTEREST_PARTS = [
        "LElbow", "RShoulder", "LShoulder",
        "RElbow", "RWrist", "LWrist"
    ]
    ANGLE_DEFINITIONS = {
        "L_elbow": ("LShoulder", "LElbow", "LWrist"),
        "R_elbow": ("RShoulder", "RElbow", "RWrist"),
        "L_shoulder": ("MidHip", "LShoulder", "LElbow"),
        "R_shoulder": ("MidHip", "RShoulder", "RElbow"),
    }
    def __init__(self, sw):
        self.person = sw.person
        self.sw = sw
        self.body_features = {}
        self.left_hand_features = {}
        self.right_hand_features = {}

        # --- COCO body parts ---
        for part_name in COCO_PARTS:
            body_part = self.person.get_body_part(part_name)
            if body_part is None:
                continue

            self.body_features[part_name] = Features(body_part, self.sw)

        # --- Hand parts (left & right) ---
        for side in ("L", "R"):
            for hand_part in HAND_PARTS:
                full_part_name = f"{side}_{hand_part}"
                body_part = self.person.get_body_part(full_part_name)
                if body_part is None:
                    continue
                if side == "L":
                    self.left_hand_features[hand_part] = Features(body_part, self.sw)
                else:
                    self.right_hand_features[hand_part] = Features(body_part, self.sw) 

        self.compute_features()

    def compute_features(self):
        self.motion_persistance = self.mean_motion_persistance_interest_parts()
        self.velocity = self.mean_velocity_interest_parts()
        self.velocity_variance = self.mean_velocity_variance_interest_parts() * CORRECTION_FACTOR_VAR
        self.max_energy = self.max_energy_interest_parts() * CORRECTION_FACTOR_HANDS
        self.left_hand_energy = self.max_hand_energy("L") * CORRECTION_FACTOR_HANDS
        self.right_hand_energy = self.max_hand_energy("R") * CORRECTION_FACTOR_HANDS
        angle_features = self.compute_joint_angles_and_velocity()
        self.l_elbow_angle = angle_features["L_elbow_angle_mean"]
        self.r_elbow_angle = angle_features["R_elbow_angle_mean"]
        self.r_shoulder_angle = angle_features["R_shoulder_angle_mean"]
        self.l_shoulder_angle = angle_features["L_shoulder_angle_mean"]
        self.l_elbow_angular_velocity = angle_features["L_elbow_angvel_mean"]
        self.r_elbow_angular_velocity = angle_features["R_elbow_angvel_mean"]
        self.r_shoulder_angular_velocity = angle_features["R_shoulder_angvel_mean"]
        self.l_shoulder_angular_velocity = angle_features["L_shoulder_angvel_mean"]
        self.distal_proximal_ratio = self.compute_distal_proximal_motion_ratio()
        self.max_acceleration = self.max_acceleration_interest_parts()
        self.max_angular = max(self.l_elbow_angular_velocity, self.r_elbow_angular_velocity,self.r_shoulder_angular_velocity,self.l_shoulder_angular_velocity)
        self.mean_baseline_distance = self.max_distance_from_baseline_interest_parts()
        self.max_motion_saliency = self.max_motion_saliency_interest_parts()
        self.max_burstiness = self.max_burstiness_interest_parts()
        self.max_directional_consistency = self.max_directional_consistency_interest_parts()
        self.max_path_efficiency = self.max_path_efficiency_interest_parts()
        self.max_direction_changes = self.max_direction_changes_interest_parts()

    def mean_motion_persistance_interest_parts(self):
        persistance_per_part = []
        count = 0
        for part in self.INTEREST_PARTS:
            f = self.body_features.get(part)
            if f and f.motion_persistence is not None:
                persistance_per_part.append(f.motion_persistence)
                count += 1
        return sum(persistance_per_part)/count if count > 0 else 0.0
    def max_acceleration_interest_parts(self):
        accs = []
        for part in self.INTEREST_PARTS:
            f = self.body_features.get(part)
            if f and f.max_acceleration is not None:
                accs.append(f.max_acceleration)

        if not accs:
            return None

        return float(np.nanpercentile(accs, 90))
    def mean_velocity_interest_parts(self):
        velocities = []
        count = 0
        for part in self.INTEREST_PARTS:
            f = self.body_features.get(part)
            if f and f.mean_velocity is not None:
                velocities.append(f.mean_velocity)
                count += 1
        return sum(velocities)/count if count > 0 else 0.0
    
    def mean_velocity_variance_interest_parts(self):
        velocities = []
        count = 0
        for part in self.INTEREST_PARTS:
            f = self.body_features.get(part)
            if f and f.velocity_variance is not None:
                velocities.append(f.velocity_variance)
                count += 1

        return sum(velocities)/count if count > 0 else 0.0
    
    def max_energy_interest_parts(self):
        energies = []
        count = 0
        for part in self.INTEREST_PARTS:
            f = self.body_features.get(part)
            if f and f.motion_energy is not None:
                energies.append(f.motion_energy)
                count +=1

        return float(np.nanpercentile(energies, 90))
    # def mean_hand_energy(self, side: str):
    #     features = (
    #         self.left_hand_features if side == "L"
    #         else self.right_hand_features
    #     )
    #     len_features = len(features)
    #     return sum(self._values(features, "motion_energy"))/len_features


    def max_hand_energy(self, side: str):
        features = (
            self.body_features.get("LWrist") if side == "L"
            else self.body_features.get("RWrist")
        )
        energies = []
        if features and features.motion_energy is not None:
                energies.append(features.motion_energy)

        return float(np.nanpercentile(energies, 90))
    # safe getter
    def _values(self, features_dict, attr):
        return [
            getattr(f, attr)
            for f in features_dict.values()
            if hasattr(f, attr) and getattr(f, attr) is not None
        ]
    def compute_distal_proximal_motion_ratio(self, eps=1e-5):
        distal_parts = ["RWrist", "LWrist", "RElbow", "LElbow"]
        proximal_parts = ["RShoulder", "LShoulder"]

        distal_velocities = []
        proximal_velocities = []

        for part in distal_parts:
            f = self.body_features.get(part)
            if f and f.mean_velocity is not None:
                distal_velocities.append(f.mean_velocity)

        for part in proximal_parts:
            f = self.body_features.get(part)
            if f and f.mean_velocity is not None:
                proximal_velocities.append(f.mean_velocity)

        # Compute per-joint averages instead of sums
        avg_distal = sum(distal_velocities) / max(len(distal_velocities), 1)
        avg_proximal = sum(proximal_velocities) / max(len(proximal_velocities), 1)

        # Ratio
        # ratio = math.log((avg_distal + eps) / (avg_proximal + eps))
        ratio = avg_distal - avg_proximal
        return ratio
    
    def compute_joint_angles_and_velocity(self, fps=24):
            """
            Compute mean joint angles and mean angular velocities over a window.

            Returns:
                features: dict with keys:
                    "{joint}_angle_mean": mean angle in radians
                    "{joint}_angvel_mean": mean angular velocity in radians/sec
            """
            np.seterr(all='ignore')
            angles = {name: [] for name in self.ANGLE_DEFINITIONS}

            # 1. Compute per-frame angles
            for frame_idx in range(self.sw.start_frame + 1, self.sw.end_frame + 1):
                for name, (p1, p2, p3) in self.ANGLE_DEFINITIONS.items():
                    body_part1 = self.person.get_body_part(p1)
                    body_part2 = self.person.get_body_part(p2)
                    body_part3 = self.person.get_body_part(p3)

                    try:
                        # Get normalized coordinates
                        p1_xy = body_part1.get_normalized_coordinates(frame_idx)
                        p2_xy = body_part2.get_normalized_coordinates(frame_idx)
                        p3_xy = body_part3.get_normalized_coordinates(frame_idx)
                        # if p1_xy is not None and p2_xy is not None and p3_xy is not None:
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
    def max_distance_from_baseline_interest_parts(self):
        """
        Computes mean distance from baseline for INTEREST_PARTS
        using the baseline corresponding to the sliding window start frame.
        """
        interest_parts = ["LWrist","RWrist"]
        distances = []

        # for part_name in self.INTEREST_PARTS:
        for part_name in interest_parts:
            body_part = self.person.get_body_part(part_name)
            if body_part is None:
                continue

            if not hasattr(body_part, "baselines"):
                continue

            

            # compute distances within window
            part_dists = []
            for frame_idx in range(self.sw.start_frame, self.sw.end_frame + 1):
                # frame = body_part.frames.get(frame_idx)
                baseline = body_part.baselines.get(frame_idx)
                if baseline is None:
                    continue

                bx, by = baseline
                x_normalized, y_normalized = body_part.get_normalized_coordinates(frame_idx)
                if x_normalized is None or y_normalized is None:
                    continue

                dx = x_normalized - bx
                dy = y_normalized - by
                part_dists.append(np.hypot(dx, dy))

            if part_dists:
                distances.append(np.percentile(part_dists, 90))

        if distances:
            return max(distances) 
        return 0.0
    
    def max_motion_saliency_interest_parts(self):
        sal = []
        for part in self.INTEREST_PARTS:
            f = self.body_features.get(part)
            if f and f.motion_saliency is not None:
                sal.append(f.motion_saliency)

        if not sal:
            return None

        return float(np.nanpercentile(sal, 90))    

    def max_burstiness_interest_parts(self):
        bursts = []
        for part in self.INTEREST_PARTS:
            f = self.body_features.get(part)
            if f and f.burstiness is not None:
                bursts.append(f.burstiness)

        if not bursts:
            return None

        return float(np.nanpercentile(bursts, 90))   
    
    def max_directional_consistency_interest_parts(self):
        dirs = []
        for part in self.INTEREST_PARTS:
            f = self.body_features.get(part)
            if f and f.directional_consistency is not None:
                dirs.append(f.directional_consistency)

        if not dirs:
            return None

        return float(np.nanpercentile(dirs, 90))   
    
    def max_direction_changes_interest_parts(self):
        dirs = []
        for part in self.INTEREST_PARTS:
            f = self.body_features.get(part)
            if f and f.direction_changes is not None:
                dirs.append(f.direction_changes)

        if not dirs:
            return None

        return float(np.nanpercentile(dirs, 90))   
    
    def max_path_efficiency_interest_parts(self):
        effs = []
        for part in self.INTEREST_PARTS:
            f = self.body_features.get(part)
            if f and f.path_efficiency is not None:
                effs.append(f.path_efficiency)

        if not effs:
            return None

        return float(np.nanpercentile(effs, 90))   