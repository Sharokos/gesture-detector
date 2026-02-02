import numpy as np

EPS = 1e-8

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
