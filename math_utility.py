import numpy as np

EPS = 1e-8


def remove_outliers_mad(values, k=3.5):
    if len(values) < 3:
        return values

    median = sorted(values)[len(values) // 2]
    deviations = [abs(v - median) for v in values]
    mad = sorted(deviations)[len(deviations) // 2]

    if mad == 0:
        return values

    filtered = []
    for v in values:
        score = abs(v - median) / mad
        filtered.append(v if score <= k else median)

    return filtered

def to_vec(a, b):
    """Vector from a -> b"""
    np.seterr(all='ignore')
    return np.array(b) - np.array(a)

def vector_norm(v):
    np.seterr(all='ignore')
    return np.linalg.norm(v) + EPS

def angle_between(v1, v2):
    np.seterr(all='ignore')
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

def smooth_signal(arr, window=3):
    np.seterr(all='ignore')
    out = arr.copy()
    for i in range(window, len(arr)):
        vals = arr[i-window:i+1]
        if not np.all(np.isnan(vals)):
            out[i] = np.nanmean(vals)
    return out

# TODO: try this smooth function and observe how the coordinates behave afterwards.
def smooth_keypoints(coords, window=3):
    np.seterr(all='ignore')
    smoothed = coords.copy()
    for i in range(len(coords)):
        start = max(0, i-window)
        end = min(len(coords), i+window+1)
        smoothed[i] = np.nanmean(coords[start:end], axis=0)
    return smoothed