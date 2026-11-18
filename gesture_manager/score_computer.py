import numpy as np

from config import (
    FEATURE_NORMALIZATION,
    LOG_FEATURES,
    NORMALIZED_CLIP,
    WEIGHTS,
    WEIGHTS_NORMALIZED,
)


def _safe(value, default=0.0):
    if value is None:
        return default
    if isinstance(value, (float, np.floating)) and np.isnan(value):
        return default
    return value


def compute_raw_score(fm):
    """Old score: raw feature values times WEIGHTS (used when FEATURE_NORMALIZATION = "none")."""
    max_angular_velocity = max(
        _safe(fm.l_elbow_angular_velocity),
        _safe(fm.r_elbow_angular_velocity),
        _safe(fm.r_shoulder_angular_velocity),
        _safe(fm.l_shoulder_angular_velocity),
    )
    return (
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


def _active_features():
    return [name for name, w in WEIGHTS_NORMALIZED.items() if w != 0]


def _transform(name, values, offset):
    values = np.asarray(values, dtype=float)
    if name in LOG_FEATURES:
        return np.log(np.maximum(values, 0.0) + offset)
    return values


def compute_normalization_stats(features_managers):
    """
    Per-video statistics for the robust normalization: for each weighted feature the median and
    interquartile range (after the log transform for LOG_FEATURES).
    """
    stats = {}
    for name in _active_features():
        raw = np.array([float(_safe(getattr(fm, name, None))) for fm in features_managers])
        if len(raw) == 0:
            continue
        # Offset keeps log() finite for windows without motion; 10% of the median motion level
        positive = raw[raw > 0]
        offset = 0.1 * float(np.median(positive)) if len(positive) else 1e-6
        offset = max(offset, 1e-9)
        values = _transform(name, raw, offset)
        median = float(np.median(values))
        q25, q75 = np.percentile(values, [25, 75])
        scale = float(q75 - q25)
        if scale <= 1e-12:
            scale = float(np.std(values)) or 1.0
        stats[name] = (offset, median, scale)
    return stats


def normalized_features(fm, stats):
    """Dict feature -> normalized value (clipped robust z-score)."""
    out = {}
    for name, (offset, median, scale) in stats.items():
        v = float(_transform(name, [_safe(getattr(fm, name, None))], offset)[0])
        out[name] = float(np.clip((v - median) / scale, *NORMALIZED_CLIP))
    return out


def compute_score(fm, stats=None):
    """
    Window score. With FEATURE_NORMALIZATION = "robust" and per-video `stats`, it is the weighted
    sum of normalized features (0 = a typical window of this video, 1 = one IQR more movement).
    Without stats (e.g. while the windows are being created), the raw score is returned.
    """
    if FEATURE_NORMALIZATION == "none" or stats is None:
        return compute_raw_score(fm)
    z = normalized_features(fm, stats)
    return sum(WEIGHTS_NORMALIZED[name] * value for name, value in z.items())
