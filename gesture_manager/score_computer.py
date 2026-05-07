import numpy as np
import math
from config import WEIGHTS
from gesture_manager.features_manager import FeaturesManager


def _safe(value, default=0.0):
    if value is None:
        return default
    if isinstance(value, (float, np.floating)) and np.isnan(value):
        return default
    return value


def compute_score(fm: FeaturesManager):
    np.seterr(all='ignore')
    max_angular_velocity = max(
        _safe(fm.l_elbow_angular_velocity),
        _safe(fm.r_elbow_angular_velocity),
        _safe(fm.r_shoulder_angular_velocity),
        _safe(fm.l_shoulder_angular_velocity),
    )

    score = (
        WEIGHTS["motion_energy_weight"] * _safe(fm.max_energy) +
        WEIGHTS["mean_velocity_weight"] * _safe(fm.velocity) +
        WEIGHTS["mean_velocity_variance_weight"] * _safe(fm.velocity_variance) +
        WEIGHTS["distal_proximal_weight"] * abs(_safe(fm.distal_proximal_ratio)) +
        WEIGHTS["persistence_weight"] * _safe(fm.motion_persistance) +
        WEIGHTS["max_angular_velocity_weight"] * max_angular_velocity +
        WEIGHTS["hands_energy_weight"] * max(
            _safe(fm.left_hand_energy),
            _safe(fm.right_hand_energy),
        ) +
        WEIGHTS["acc_weight"] * _safe(fm.max_acceleration) +
        WEIGHTS["saliency_weight"] * _safe(fm.max_motion_saliency) +
        WEIGHTS["burst_weight"] * _safe(fm.max_burstiness) +
        WEIGHTS["changes_weight"] * _safe(fm.max_direction_changes) +
        WEIGHTS["path_efficiency_weight"] * _safe(fm.max_path_efficiency)
    )

    return score