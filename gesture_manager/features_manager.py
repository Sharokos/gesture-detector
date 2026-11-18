import numpy as np

from config import COCO_PARTS, CORRECTION_FACTOR_HANDS, CORRECTION_FACTOR_VAR
from gesture_manager.features import Features

# Virtual reference point: straight down from the joint. Used for the shoulder angle so that it does
# not depend on the hips being visible (a table often hides them for a seated person).
DOWN = "__down__"


def _p90(values):
    values = [v for v in values if v is not None]
    if not values:
        return None
    return float(np.nanpercentile(values, 90))


class FeaturesManager:
    INTEREST_PARTS = [
        "LElbow", "RShoulder", "LShoulder",
        "RElbow", "RWrist", "LWrist"
    ]
    ANGLE_DEFINITIONS = {
        "L_elbow": ("LShoulder", "LElbow", "LWrist"),
        "R_elbow": ("RShoulder", "RElbow", "RWrist"),
        # Upper arm vs. vertical (arm raising / swinging)
        "L_shoulder": (DOWN, "LShoulder", "LElbow"),
        "R_shoulder": (DOWN, "RShoulder", "RElbow"),
    }
    BASELINE_PARTS = ["LWrist", "RWrist"]

    def __init__(self, sw):
        self.person = sw.person
        self.sw = sw
        self.body_features = {}
        # Hand keypoints are not used by the score; skipping their per-window features is a large
        # speed-up. The raw hand data is still available on the person if needed.
        self.left_hand_features = {}
        self.right_hand_features = {}

        for part_name in COCO_PARTS:
            body_part = self.person.get_body_part(part_name)
            if body_part is not None:
                self.body_features[part_name] = Features(body_part, self.sw)

        self.compute_features()

    def _interest(self, attr):
        out = []
        for part in self.INTEREST_PARTS:
            f = self.body_features.get(part)
            if f is not None and getattr(f, attr) is not None:
                out.append(getattr(f, attr))
        return out

    def _mean_interest(self, attr):
        values = self._interest(attr)
        return sum(values) / len(values) if values else 0.0

    def compute_features(self):
        self.motion_persistance = self._mean_interest("motion_persistence")
        self.velocity = self._mean_interest("mean_velocity")
        self.velocity_variance = self._mean_interest("velocity_variance") * CORRECTION_FACTOR_VAR
        self.max_energy = (_p90(self._interest("motion_energy")) or 0.0) * CORRECTION_FACTOR_HANDS
        self.left_hand_energy = self._wrist_energy("LWrist") * CORRECTION_FACTOR_HANDS
        self.right_hand_energy = self._wrist_energy("RWrist") * CORRECTION_FACTOR_HANDS

        angle_features = self.compute_joint_angles_and_velocity()
        self.l_elbow_angle = angle_features["L_elbow_angle_mean"]
        self.r_elbow_angle = angle_features["R_elbow_angle_mean"]
        self.r_shoulder_angle = angle_features["R_shoulder_angle_mean"]
        self.l_shoulder_angle = angle_features["L_shoulder_angle_mean"]
        self.l_elbow_angular_velocity = angle_features["L_elbow_angvel_mean"]
        self.r_elbow_angular_velocity = angle_features["R_elbow_angvel_mean"]
        self.r_shoulder_angular_velocity = angle_features["R_shoulder_angvel_mean"]
        self.l_shoulder_angular_velocity = angle_features["L_shoulder_angvel_mean"]
        self.max_angular = float(np.nanmax([
            self.l_elbow_angular_velocity, self.r_elbow_angular_velocity,
            self.r_shoulder_angular_velocity, self.l_shoulder_angular_velocity,
            0.0,
        ]))

        self.distal_proximal_ratio = self.compute_distal_proximal_motion_ratio()
        self.max_acceleration = _p90(self._interest("max_acceleration"))
        self.max_motion_saliency = (_p90(self._interest("motion_saliency")) or 0.0) * 0.1
        self.max_burstiness = _p90(self._interest("burstiness"))
        self.max_directional_consistency = _p90(self._interest("directional_consistency"))
        self.max_path_efficiency = (_p90(self._interest("path_efficiency")) or 0.0) * 0.1
        self.max_direction_changes = _p90(self._interest("direction_changes"))

        dist, dist_x, dist_y = self.distance_from_baseline()
        self.max_distance_horizontal = dist_x
        self.max_distance_vertical = dist_y
        # How far the wrists are from their rest position (used to detect "holds")
        self.mean_baseline_distance = dist

    def _wrist_energy(self, part):
        f = self.body_features.get(part)
        return f.motion_energy if f is not None else 0.0

    def compute_distal_proximal_motion_ratio(self):
        distal = [self.body_features[p].mean_velocity for p in ("RWrist", "LWrist", "RElbow", "LElbow")
                  if p in self.body_features]
        proximal = [self.body_features[p].mean_velocity for p in ("RShoulder", "LShoulder")
                    if p in self.body_features]
        avg_distal = sum(distal) / max(len(distal), 1)
        avg_proximal = sum(proximal) / max(len(proximal), 1)
        return avg_distal - avg_proximal

    def _coords(self, part_name, start, end):
        part = self.person.get_body_part(part_name)
        return np.stack([part.xn[start:end], part.yn[start:end]], axis=1)

    def compute_joint_angles_and_velocity(self):
        """Mean joint angle (rad) and mean absolute angular velocity (rad/s) over the window."""
        start, end = self.sw.start_frame + 1, self.sw.end_frame + 1
        features = {}
        with np.errstate(all="ignore"):
            for name, (p1, p2, p3) in self.ANGLE_DEFINITIONS.items():
                b = self._coords(p2, start, end)
                c = self._coords(p3, start, end)
                if p1 == DOWN:
                    a = b + np.array([0.0, 1.0])  # image y axis points down
                else:
                    a = self._coords(p1, start, end)
                v1, v2 = a - b, c - b
                cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
                dot = (v1 * v2).sum(axis=1)
                angle = np.abs(np.arctan2(cross, dot))

                finite = np.isfinite(angle)
                features[f"{name}_angle_mean"] = float(np.mean(angle[finite])) if finite.any() else np.nan
                if len(angle) > 1:
                    ang_vel = np.abs(np.gradient(angle)) * self.sw.fps
                    ok = np.isfinite(ang_vel)
                    features[f"{name}_angvel_mean"] = float(np.mean(ang_vel[ok])) if ok.any() else np.nan
                else:
                    features[f"{name}_angvel_mean"] = np.nan
        return features

    def distance_from_baseline(self):
        """
        90th percentile (over the window) of the wrist distance to its rest position, max over both
        wrists. Returns (euclidean, |horizontal|, |vertical|).
        """
        start, end = self.sw.start_frame, self.sw.end_frame + 1
        dist, dist_x, dist_y = [], [], []
        for part_name in self.BASELINE_PARTS:
            part = self.person.get_body_part(part_name)
            if part is None or len(part.baseline_x) == 0:
                continue
            dx = part.xn[start:end] - part.baseline_x[start:end]
            dy = part.yn[start:end] - part.baseline_y[start:end]
            ok = np.isfinite(dx) & np.isfinite(dy)
            if not ok.any():
                continue
            dx, dy = dx[ok], dy[ok]
            dist.append(np.percentile(np.hypot(dx, dy), 90))
            dist_x.append(np.percentile(np.abs(dx), 90))
            dist_y.append(np.percentile(np.abs(dy), 90))
        if not dist:
            return 0.0, 0.0, 0.0
        return float(max(dist)), float(max(dist_x)), float(max(dist_y))
