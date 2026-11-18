import numpy as np

EPS = 1e-8


def remove_outliers_mad(values, k=3.5):
    """
    Clip (winsorize) values further than k * MAD from the median to that bound.

    Previously outliers were REPLACED by the median, which turned the strongest gestures into
    "rest" windows. Clipping keeps them on the right side of the threshold while still limiting the
    influence of OpenPose glitches.
    """
    values = np.asarray(values, dtype=float)
    if len(values) < 3:
        return values.tolist()
    finite = np.isfinite(values)
    if not finite.any():
        return values.tolist()
    median = np.median(values[finite])
    mad = np.median(np.abs(values[finite] - median))
    if mad == 0:
        return values.tolist()
    clipped = np.clip(values, median - k * mad, median + k * mad)
    return clipped.tolist()


def otsu_threshold(values, bins=256):
    """Otsu's threshold on a 1-D sample: the split that maximises the between-class variance."""
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) < 2 or values.min() == values.max():
        raise ValueError("Not enough spread for Otsu thresholding")
    hist, edges = np.histogram(values, bins=bins)
    centers = (edges[:-1] + edges[1:]) / 2
    weight1 = np.cumsum(hist)
    weight2 = np.cumsum(hist[::-1])[::-1]
    mean1 = np.cumsum(hist * centers) / np.maximum(weight1, 1)
    mean2 = (np.cumsum((hist * centers)[::-1]) / np.maximum(weight2[::-1], 1))[::-1]
    between = weight1[:-1] * weight2[1:] * (mean1[:-1] - mean2[1:]) ** 2
    return float(centers[:-1][np.argmax(between)])


def log_otsu_threshold(values):
    """Otsu in log space: motion scores are heavily skewed (long tail of strong movements)."""
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values) & (values > 0)]
    return float(np.exp(otsu_threshold(np.log(values))))


def to_vec(a, b):
    """Vector from a -> b"""
    return np.array(b) - np.array(a)


def vector_norm(v):
    return np.linalg.norm(v) + EPS


def angle_between(v1, v2):
    """Angle in radians between vectors"""
    v1n = v1 / vector_norm(v1)
    v2n = v2 / vector_norm(v2)
    dot = np.clip(np.dot(v1n, v2n), -1.0, 1.0)
    return np.arccos(dot)


def joint_angle(p1, p2, p3):
    """
    Angle at p2 formed by p1-p2-p3
    Example: shoulder–elbow–wrist
    """
    v1 = to_vec(p2, p1)
    v2 = to_vec(p2, p3)
    return angle_between(v1, v2)


def euclidean_distance(p1, p2):
    return vector_norm(to_vec(p1, p2))


def smooth_keypoints(coords, window=3):
    """Centered moving average over [i - window, i + window], ignoring NaN."""
    from data_model.body_part import centered_nanmean

    return centered_nanmean(np.asarray(coords, dtype=float), window)
