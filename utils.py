import numpy as np
from collections import deque
from sklearn.linear_model import LinearRegression

from difflib import SequenceMatcher


def is_raw_count_stable(counts: deque, stable_frames: int, tolerance: int) -> bool:
    """Check if raw counts are stable within tolerance"""
    if len(counts) < stable_frames:
        return False
    window = list(counts)[-stable_frames:]
    return (max(window) - min(window)) <= tolerance


def rotate_points(points: np.ndarray, angle: float) -> np.ndarray:
    """Rotate 2D points by angle in radians"""
    R = np.array([
        [np.cos(angle), -np.sin(angle)],
        [np.sin(angle), np.cos(angle)]
    ])
    return (R @ points.T).T


def cluster_1d(values, tol=20):
    """Cluster 1D values with tolerance"""
    values = sorted(values)
    groups, current = [], [values[0]]
    for v in values[1:]:
        if abs(v - np.mean(current)) < tol:
            current.append(v)
        else:
            groups.append(current)
            current = [v]
    groups.append(current)
    return groups

def plate_similarity(p1: str, p2: str) -> float:
    """
    Returns similarity ratio between 2 plates.
    1.0 = exact match
    0.0 = completely different
    """
    return SequenceMatcher(None, p1, p2).ratio()