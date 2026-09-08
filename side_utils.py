#!/usr/bin/env python3
"""
Side geometry utilities – implements the refined row/column detection
from app1_refined_stable_v4.py, plus a new lightweight, robust geometry
function for production counting.

All functions operate on original image coordinates.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression

logger = logging.getLogger(__name__)

# =============================================================================
# Configuration
# =============================================================================

@dataclass(frozen=True)
class RowConfig:
    max_rows: int = 3
    kmeans_n_init: int = 20
    min_points_for_fit: int = 2


@dataclass(frozen=True)
class ColumnConfig:
    max_columns_hard: int = 17
    max_centers_per_column: int = 3
    last_column_max_centers: int = 2

    # Column construction
    x_scale: float = 4.0
    gap_multiplier: float = 4.0
    threshold_min_px: float = 36.0
    threshold_max_px: float = 140.0
    ema_alpha: float = 0.72

    # Re-clustering
    recluster_scales: Tuple[float, ...] = (1.0, 0.88, 1.12, 0.76, 1.24)
    max_recluster_rounds: int = 20

    # Capacity repair
    split_keep_count: int = 3
    terminal_move_attempts: int = 10


@dataclass(frozen=True)
class LineConfig:
    # Angle deviation from vertical (degrees)
    leading_max_dev_deg: float = 5.0
    trailing_max_dev_deg: float = 10.0
    middle_max_dev_deg: float = 7.0

    leading_column_count: int = 5
    trailing_column_count: int = 5

    candidate_span: float = 0.18
    candidate_steps: int = 61
    slope_penalty_weight: float = 0.25
    box_penalty_weight: float = 40.0
    box_margin_px: float = 6.0

    repair_iterations: int = 2
    repair_min_gap_px: float = 8.0
    sample_y_count: int = 12
    repair_sample_weight: float = 1.35


ROW_CFG = RowConfig()
COL_CFG = ColumnConfig()
LINE_CFG = LineConfig()


# =============================================================================
# Helpers
# =============================================================================

def _safe_int(v) -> int:
    return int(round(float(v)))


def boxes_to_centers(boxes: np.ndarray) -> np.ndarray:
    if boxes is None or len(boxes) == 0:
        return np.empty((0, 2), dtype=float)
    boxes = np.asarray(boxes, dtype=float)
    return np.stack(
        [(boxes[:, 0] + boxes[:, 2]) / 2.0, (boxes[:, 1] + boxes[:, 3]) / 2.0],
        axis=1,
    )


def _counts_from_ids(ids: np.ndarray, n_groups: int) -> List[int]:
    if ids is None or len(ids) == 0 or n_groups <= 0:
        return []
    return [int(np.sum(ids == i)) for i in range(n_groups)]


def _pava_non_decreasing(values: np.ndarray, weights: Optional[np.ndarray] = None) -> np.ndarray:
    """Pool adjacent violators algorithm for isotonic regression."""
    v = np.asarray(values, dtype=float)
    n = len(v)
    if n == 0:
        return v

    if weights is None:
        w = np.ones(n, dtype=float)
    else:
        w = np.asarray(weights, dtype=float).copy()

    block_values = v.tolist()
    block_weights = w.tolist()
    block_sizes = [1] * n

    i = 0
    while i < len(block_values) - 1:
        if block_values[i] <= block_values[i + 1]:
            i += 1
            continue

        total_w = block_weights[i] + block_weights[i + 1]
        merged = (
            block_values[i] * block_weights[i] + block_values[i + 1] * block_weights[i + 1]
        ) / total_w

        block_values[i] = merged
        block_weights[i] = total_w
        block_sizes[i] += block_sizes[i + 1]

        del block_values[i + 1]
        del block_weights[i + 1]
        del block_sizes[i + 1]

        if i > 0:
            i -= 1

    out = np.empty(n, dtype=float)
    pos = 0
    for val, size in zip(block_values, block_sizes):
        out[pos:pos + size] = val
        pos += size
    return out


def _project_strictly_increasing(xs: np.ndarray, min_gap: float) -> np.ndarray:
    xs = np.asarray(xs, dtype=float)
    if len(xs) <= 1:
        return xs.copy()
    z = xs - np.arange(len(xs), dtype=float) * float(min_gap)
    z_proj = _pava_non_decreasing(z)
    return z_proj + np.arange(len(xs), dtype=float) * float(min_gap)


def _line_x_at_y(slope: float, intercept: float, y: float) -> float:
    return float(slope * y + intercept)


def _line_deviation_from_vertical_deg(slope: float) -> float:
    return float(np.degrees(np.arctan(abs(float(slope)))))


# =============================================================================
# Row fitting (new logic)
# =============================================================================

def compute_row_lines(
    centers: np.ndarray,
    max_rows: int = ROW_CFG.max_rows,
) -> Tuple[List[Tuple[float, float]], np.ndarray]:
    """Cluster rows using KMeans on Y (no rotation)."""
    if centers is None or len(centers) == 0:
        return [], np.array([], dtype=int)

    centers = np.asarray(centers, dtype=float)

    if len(centers) == 1:
        return [(0.0, float(centers[0, 1]))], np.array([0], dtype=int)

    n_clusters = min(int(max_rows), len(centers))

    kmeans = KMeans(
        n_clusters=n_clusters,
        random_state=0,
        n_init=ROW_CFG.kmeans_n_init,
    )
    labels = kmeans.fit_predict(centers[:, 1].reshape(-1, 1))

    order = np.argsort(kmeans.cluster_centers_.flatten())

    row_lines: List[Tuple[float, float]] = []
    remap: Dict[int, int] = {}

    for new_idx, cluster_idx in enumerate(order):
        remap[int(cluster_idx)] = int(new_idx)

        pts = centers[labels == cluster_idx]

        if len(pts) >= ROW_CFG.min_points_for_fit:
            reg = LinearRegression().fit(pts[:, 0].reshape(-1, 1), pts[:, 1])
            slope = float(reg.coef_[0])
            intercept = float(reg.intercept_)
        else:
            slope = 0.0
            intercept = float(pts[0, 1])

        row_lines.append((slope, intercept))

    row_ids = np.array([remap[int(l)] for l in labels], dtype=int)
    return row_lines, row_ids


def _row_mean_y(centers: np.ndarray, row_ids: np.ndarray) -> Dict[int, float]:
    out: Dict[int, float] = {}
    for rid in np.unique(row_ids):
        pts = centers[row_ids == rid]
        if len(pts):
            out[int(rid)] = float(np.mean(pts[:, 1]))
    return out


def _bottom_row_id(row_ids: Optional[np.ndarray], centers: np.ndarray) -> Optional[int]:
    if row_ids is None or len(row_ids) == 0:
        return None
    mean_y = _row_mean_y(centers, row_ids)
    if not mean_y:
        return None
    return max(mean_y, key=mean_y.get)


def _top_row_id(row_ids: Optional[np.ndarray], centers: np.ndarray) -> Optional[int]:
    if row_ids is None or len(row_ids) == 0:
        return None
    mean_y = _row_mean_y(centers, row_ids)
    if not mean_y:
        return None
    return min(mean_y, key=mean_y.get)


# =============================================================================
# Column clustering (new logic)
# =============================================================================

def estimate_column_threshold(xs: np.ndarray) -> float:
    if len(xs) < 2:
        return 80.0
    sx = np.sort(np.asarray(xs, dtype=float))
    gaps = np.diff(sx)
    if len(gaps) == 0:
        return 80.0
    median_gap = float(np.median(gaps))
    return float(np.clip(COL_CFG.gap_multiplier * median_gap,
                         COL_CFG.threshold_min_px,
                         COL_CFG.threshold_max_px))


def _match_sorted_points_to_columns(row_xs: np.ndarray, column_xs: np.ndarray) -> np.ndarray:
    row_xs = np.asarray(row_xs, dtype=float)
    column_xs = np.asarray(column_xs, dtype=float)

    m = len(row_xs)
    n = len(column_xs)
    if m == 0 or n == 0:
        return np.array([], dtype=int)
    if m == 1:
        return np.array([int(np.argmin(np.abs(column_xs - row_xs[0])))], dtype=int)

    # DP matching for m > n case (more points than columns)
    if m > n:
        assignments = np.full(m, -1, dtype=int)
        available = list(range(n))
        last = -1
        for i, x in enumerate(row_xs):
            candidate_cols = [j for j in available if j > last]
            if not candidate_cols:
                candidate_cols = list(range(last + 1, n))
            if not candidate_cols:
                candidate_cols = list(range(n))
            j = min(candidate_cols, key=lambda jj: abs(column_xs[jj] - x))
            assignments[i] = int(j)
            if j in available:
                available.remove(j)
            last = max(last, int(j))
        return assignments

    # DP for m <= n
    dp = np.full((m + 1, n + 1), np.inf, dtype=float)
    take = np.zeros((m + 1, n + 1), dtype=np.uint8)
    dp[0, :] = 0.0

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            skip_cost = dp[i, j - 1]
            match_cost = dp[i - 1, j - 1] + abs(float(row_xs[i - 1]) - float(column_xs[j - 1]))
            if match_cost <= skip_cost:
                dp[i, j] = match_cost
                take[i, j] = 1
            else:
                dp[i, j] = skip_cost
                take[i, j] = 0

    assignments = np.full(m, -1, dtype=int)
    i, j = m, n
    while i > 0 and j > 0:
        if take[i, j] == 1:
            assignments[i - 1] = j - 1
            i -= 1
            j -= 1
        else:
            j -= 1

    # Fallback: fill any unassigned
    if np.any(assignments < 0):
        used = set()
        last = -1
        for i, x in enumerate(row_xs):
            if assignments[i] >= 0:
                used.add(int(assignments[i]))
                last = int(assignments[i])
                continue
            candidate_cols = [jj for jj in range(n) if jj not in used and jj > last]
            if not candidate_cols:
                candidate_cols = [jj for jj in range(n) if jj not in used]
            if not candidate_cols:
                candidate_cols = list(range(n))
            j_best = min(candidate_cols, key=lambda jj: abs(column_xs[jj] - x))
            assignments[i] = int(j_best)
            used.add(int(j_best))
            last = int(j_best)

    return assignments


def base_threshold_for_row(row_xs: np.ndarray) -> float:
    if len(row_xs) < 2:
        return 80.0
    return estimate_column_threshold(row_xs)


def _initial_column_ids(
    centers: np.ndarray,
    row_ids: Optional[np.ndarray],
    is_left: bool,
    max_centers_per_column: int,
    threshold_scale: float,
) -> np.ndarray:
    """
    Build initial column assignments using a row‑aware anchor method.
    The anchor row is the most populated row (by number of detections).
    """
    n = len(centers)
    if n == 0:
        return np.array([], dtype=int)

    centers = np.asarray(centers, dtype=float)
    work_x = centers[:, 0] * float(COL_CFG.x_scale)

    if row_ids is None or len(row_ids) != n:
        # Fallback: simple greedy clustering
        order = np.argsort(work_x)
        xs = work_x[order]
        base_thr = estimate_column_threshold(xs) * float(threshold_scale)

        groups: List[List[int]] = [[int(order[0])]]
        group_xs: List[List[float]] = [[float(xs[0])]]
        group_rows: List[set] = [set()]

        for k in range(1, n):
            idx = int(order[k])
            x = float(work_x[idx])
            gap = abs(x - float(xs[k - 1]))
            same_row = False
            local_rows = set()
            if row_ids is not None and len(row_ids) == n:
                local_rows = {int(row_ids[idx])}
                same_row = len(group_rows[-1].intersection(local_rows)) > 0

            fits = (gap <= base_thr) and (len(groups[-1]) < max_centers_per_column) and (not same_row)
            if fits:
                groups[-1].append(idx)
                group_xs[-1].append(x)
                group_rows[-1].update(local_rows)
            else:
                groups.append([idx])
                group_xs.append([x])
                group_rows.append(set(local_rows))

        group_mean_x = [float(np.mean(g)) for g in group_xs]
        phys_order = np.argsort(group_mean_x)
        remap = {int(old): int(new) for new, old in enumerate(phys_order)}

        column_ids = np.zeros(n, dtype=int)
        for gid, members in enumerate(groups):
            new_id = remap[gid]
            for idx in members:
                column_ids[idx] = int(new_id)
        return column_ids

    # Row‑aware construction
    row_ids = np.asarray(row_ids, dtype=int)
    unique_rows = sorted(np.unique(row_ids).tolist())
    if not unique_rows:
        return np.zeros(n, dtype=int)

    row_mean_y = _row_mean_y(centers, row_ids)
    row_to_indices = {int(rid): np.where(row_ids == rid)[0] for rid in unique_rows}

    # Pick anchor row: most populated, tie by lower Y
    candidate_rows = []
    for rid in unique_rows:
        idx = row_to_indices[int(rid)]
        candidate_rows.append((len(idx), row_mean_y[int(rid)], int(rid)))
    candidate_rows.sort(reverse=True)
    anchor_row = candidate_rows[0][2]

    anchor_idx = row_to_indices[int(anchor_row)]
    anchor_order = anchor_idx[np.argsort(work_x[anchor_idx])]

    column_xs = [float(work_x[i]) for i in anchor_order]
    assignments = np.full(n, -1, dtype=int)

    for col_idx, pt_idx in enumerate(anchor_order):
        assignments[int(pt_idx)] = int(col_idx)

    others = [rid for rid in unique_rows if int(rid) != int(anchor_row)]
    others = sorted(others, key=lambda rr: (-row_mean_y.get(int(rr), -np.inf),
                                            len(row_to_indices[int(rr)])))

    for rid in others:
        idx = row_to_indices[int(rid)]
        row_order = idx[np.argsort(work_x[idx])]
        row_xs = work_x[row_order]

        if len(row_xs) == 0:
            continue

        # Insert new columns if this row has more points than current columns
        while len(row_xs) > len(column_xs) and len(column_xs) < COL_CFG.max_columns_hard:
            gaps = np.diff(row_xs)
            if len(gaps) == 0:
                break
            gap_idx = int(np.argmax(gaps))
            gap_val = float(gaps[gap_idx])
            if gap_val < base_threshold_for_row(row_xs) * float(threshold_scale):
                break
            new_x = float((row_xs[gap_idx] + row_xs[gap_idx + 1]) / 2.0)
            insert_at = int(np.searchsorted(np.asarray(column_xs, dtype=float), new_x))
            column_xs.insert(insert_at, new_x)
            assignments[assignments >= insert_at] += 1

        current_cols = np.asarray(column_xs, dtype=float)
        base_thr = base_threshold_for_row(row_xs) * float(threshold_scale)

        # Add extreme left/right columns if needed
        while len(column_xs) < COL_CFG.max_columns_hard:
            current_cols = np.asarray(column_xs, dtype=float)
            left_gap = current_cols[0] - row_xs[0]
            right_gap = row_xs[-1] - current_cols[-1]

            if left_gap > base_thr:
                column_xs.insert(0, float(row_xs[0]))
                assignments[assignments >= 0] += 1
                continue
            if right_gap > base_thr:
                column_xs.append(float(row_xs[-1]))
                continue
            break

        matched = _match_sorted_points_to_columns(row_xs, current_cols)

        for local_i, col_idx in enumerate(matched):
            assignments[int(row_order[local_i])] = int(col_idx)

        # Running average refinement
        for col_idx in np.unique(matched):
            col_idx = int(col_idx)
            pts_here = row_xs[matched == col_idx]
            if len(pts_here) > 0:
                column_xs[col_idx] = float(
                    COL_CFG.ema_alpha * column_xs[col_idx] +
                    (1.0 - COL_CFG.ema_alpha) * float(np.mean(pts_here))
                )

    # Re‑order columns by physical x
    phys_order = np.argsort(np.asarray(column_xs, dtype=float))
    remap = {int(old): int(new) for new, old in enumerate(phys_order)}

    column_ids = np.zeros(n, dtype=int)
    for idx, col_id in enumerate(assignments):
        if col_id < 0:
            nearest = int(np.argmin(np.abs(np.asarray(column_xs, dtype=float) - work_x[idx])))
            col_id = nearest
        column_ids[idx] = remap[int(col_id)]

    return column_ids


def split_overfull_columns(
    centers: np.ndarray,
    column_ids: np.ndarray,
    max_centers_per_column: int
) -> np.ndarray:
    if centers is None or len(centers) == 0 or column_ids is None or len(column_ids) == 0:
        return column_ids

    centers = np.asarray(centers, dtype=float)
    column_ids = np.asarray(column_ids, dtype=int).copy()

    while True:
        uniq = sorted(np.unique(column_ids).tolist())
        raw_counts = {int(cid): int(np.sum(column_ids == cid)) for cid in uniq}
        overflow = next((cid for cid in uniq if raw_counts[cid] > max_centers_per_column), None)
        if overflow is None:
            break
        if len(uniq) >= COL_CFG.max_columns_hard:
            break

        idx = np.where(column_ids == overflow)[0]
        order = np.argsort(centers[idx, 0])
        keep = min(COL_CFG.split_keep_count, len(idx) - 1)
        split_idx = idx[order[keep:]]

        column_ids[column_ids > overflow] += 1
        column_ids[split_idx] = overflow + 1

    return column_ids


def fix_terminal_capacity(
    centers: np.ndarray,
    column_ids: np.ndarray,
    row_ids: Optional[np.ndarray],
    is_left: bool,
    terminal_cap: int,
) -> np.ndarray:
    if centers is None or len(centers) == 0 or len(column_ids) == 0:
        return column_ids

    centers = np.asarray(centers, dtype=float)
    column_ids = np.asarray(column_ids, dtype=int).copy()
    row_ids = None if row_ids is None else np.asarray(row_ids, dtype=int)

    uniq = sorted(np.unique(column_ids).tolist())
    if len(uniq) < 2:
        return column_ids

    terminal = int(uniq[-1] if is_left else uniq[0])
    inward = int(uniq[-2] if is_left else uniq[1])
    bottom_rid = _bottom_row_id(row_ids, centers) if row_ids is not None else None

    attempts = 0
    while int(np.sum(column_ids == terminal)) > terminal_cap and attempts < COL_CFG.terminal_move_attempts:
        attempts += 1
        term_idx = np.where(column_ids == terminal)[0]
        if len(term_idx) == 0:
            break

        if is_left:
            ordered = term_idx[np.argsort(centers[term_idx, 0])]
        else:
            ordered = term_idx[np.argsort(-centers[term_idx, 0])]

        inward_rows = set()
        if row_ids is not None:
            inward_rows = set(row_ids[column_ids == inward].tolist())

        moved = False
        for idx in ordered:
            if row_ids is not None:
                rid = int(row_ids[idx])
                if rid == bottom_rid:
                    continue
                if rid in inward_rows:
                    continue
            column_ids[int(idx)] = inward
            moved = True
            break

        if not moved:
            break

    return column_ids


def force_terminal_column_creation(
    centers: np.ndarray,
    column_ids: np.ndarray,
    row_ids: Optional[np.ndarray],
    is_left: bool,
) -> np.ndarray:
    """
    Ensure the terminal column has at least one point (otherwise move one from neighbour).
    """
    centers = np.asarray(centers)
    column_ids = np.asarray(column_ids).copy()

    uniq = sorted(np.unique(column_ids))
    if len(uniq) < 2:
        return column_ids

    terminal = uniq[-1] if is_left else uniq[0]
    terminal_pts = centers[column_ids == terminal]
    if len(terminal_pts) >= 2:
        return column_ids

    candidate_idx = []
    for idx in range(len(centers)):
        if column_ids[idx] == terminal:
            continue
        candidate_idx.append(idx)

    if not candidate_idx:
        return column_ids

    xs = centers[candidate_idx, 0]
    if is_left:
        promote_idx = candidate_idx[np.argmax(xs)]
    else:
        promote_idx = candidate_idx[np.argmin(xs)]

    column_ids[promote_idx] = terminal
    return column_ids


# =============================================================================
# Column line fitting and validation (new logic)
# =============================================================================

def _choose_anchor_point(
    pts: np.ndarray,
    row_ids: Optional[np.ndarray],
    centers: np.ndarray,
    mask: np.ndarray,
) -> Tuple[float, float]:
    if row_ids is not None and len(row_ids) == len(centers):
        bottom_rid = _bottom_row_id(row_ids, centers)
        if bottom_rid is not None:
            bottom_mask = mask & (row_ids == bottom_rid)
            if np.any(bottom_mask):
                bottom_pts = centers[bottom_mask]
                idx = int(np.argmax(bottom_pts[:, 1]))
                return float(bottom_pts[idx, 0]), float(bottom_pts[idx, 1])

        top_rid = _top_row_id(row_ids, centers)
        if top_rid is not None:
            top_mask = mask & (row_ids == top_rid)
            if np.any(top_mask):
                top_pts = centers[top_mask]
                idx = int(np.argmin(top_pts[:, 1]))
                return float(top_pts[idx, 0]), float(top_pts[idx, 1])

    idx = int(np.argmax(pts[:, 1]))
    return float(pts[idx, 0]), float(pts[idx, 1])


def _column_angle_budget_deg(col_idx: int, n_cols: int, is_left: bool) -> float:
    if n_cols <= 1:
        return LINE_CFG.leading_max_dev_deg
    if not is_left:
        col_idx = n_cols - 1 - col_idx
    if col_idx < min(LINE_CFG.leading_column_count, n_cols):
        return LINE_CFG.leading_max_dev_deg
    if col_idx >= max(n_cols - LINE_CFG.trailing_column_count, 0):
        return LINE_CFG.trailing_max_dev_deg
    return LINE_CFG.middle_max_dev_deg


def _box_crosses_line_at_y(slope: float, intercept: float, box: np.ndarray, margin: float) -> bool:
    x1, y1, x2, y2 = map(float, box[:4])
    for yy in (y1, 0.33 * y1 + 0.67 * y2, 0.5 * (y1 + y2), 0.67 * y1 + 0.33 * y2, y2):
        x_pred = _line_x_at_y(slope, intercept, yy)
        if x1 - margin <= x_pred <= x2 + margin:
            return True
    return False


def _line_intersection_xy(line_a: Tuple[float, float], line_b: Tuple[float, float]) -> Optional[Tuple[float, float]]:
    a1, b1 = line_a
    a2, b2 = line_b
    if abs(a1 - a2) < 1e-9:
        return None
    y = (b2 - b1) / (a1 - a2)
    x = a1 * y + b1
    return float(x), float(y)


def _point_in_box(x: float, y: float, box: np.ndarray, margin: float) -> bool:
    x1, y1, x2, y2 = map(float, box[:4])
    return (x1 - margin) <= x <= (x2 + margin) and (y1 - margin) <= y <= (y2 + margin)


def _sample_y_values(centers: np.ndarray, boxes: Optional[np.ndarray]) -> np.ndarray:
    ys: List[float] = []
    if centers is not None and len(centers) > 0:
        ys.extend([float(v) for v in centers[:, 1].tolist()])
    if boxes is not None and len(boxes) > 0:
        boxes = np.asarray(boxes, dtype=float)
        ys.extend([float(v) for v in boxes[:, 1].tolist()])
        ys.extend([float(v) for v in boxes[:, 3].tolist()])
        ys.extend([float(0.5 * (b[1] + b[3])) for b in boxes])

    if not ys:
        return np.array([], dtype=float)

    ys = np.asarray(sorted(set(np.round(ys, 2).tolist())), dtype=float)
    if len(ys) <= LINE_CFG.sample_y_count:
        return ys
    return np.linspace(float(np.min(ys)), float(np.max(ys)), LINE_CFG.sample_y_count)


def _fit_line_with_anchor_and_constraints(
    pts: np.ndarray,
    anchor_xy: Tuple[float, float],
    other_boxes: Optional[np.ndarray],
    max_dev_deg: float,
    extra_samples: Optional[np.ndarray] = None,
    extra_weights: Optional[np.ndarray] = None,
) -> Tuple[float, float]:
    pts = np.asarray(pts, dtype=float)
    ax, ay = float(anchor_xy[0]), float(anchor_xy[1])

    max_slope = float(np.tan(np.deg2rad(max_dev_deg)))

    if len(pts) == 0 and (extra_samples is None or len(extra_samples) == 0):
        return 0.0, ax

    x = pts[:, 0]
    y = pts[:, 1]
    weights = np.ones(len(pts), dtype=float)

    if extra_samples is not None and len(extra_samples) > 0:
        extra_samples = np.asarray(extra_samples, dtype=float)
        if extra_weights is None:
            extra_weights = np.ones(len(extra_samples), dtype=float)
        else:
            extra_weights = np.asarray(extra_weights, dtype=float)
        x = np.concatenate([x, extra_samples[:, 0]])
        y = np.concatenate([y, extra_samples[:, 1]])
        weights = np.concatenate([weights, extra_weights])

    dy = y - ay
    dx = x - ax
    denom = float(np.sum(weights * dy * dy))
    slope0 = float(np.sum(weights * dy * dx) / denom) if denom > 1e-9 else 0.0
    slope0 = float(np.clip(slope0, -max_slope, max_slope))

    candidates = np.unique(
        np.clip(
            np.concatenate([
                np.array([0.0, slope0], dtype=float),
                slope0 + np.linspace(-LINE_CFG.candidate_span, LINE_CFG.candidate_span, LINE_CFG.candidate_steps),
            ]),
            -max_slope,
            max_slope,
        )
    )

    best_slope = slope0
    best_score = np.inf

    if other_boxes is not None and len(other_boxes) > 0:
        other_boxes = np.asarray(other_boxes, dtype=float)

    for slope in candidates:
        intercept = ax - slope * ay
        preds = slope * y + intercept
        fit_err = float(np.average(np.abs(preds - x), weights=weights))

        box_pen = 0.0
        if other_boxes is not None and len(other_boxes) > 0:
            for box in other_boxes:
                if box is None or len(box) < 4:
                    continue
                if _box_crosses_line_at_y(slope, intercept, box, LINE_CFG.box_margin_px):
                    box_pen += 1.0

        slope_pen = LINE_CFG.slope_penalty_weight * abs(float(slope) - slope0)
        score = fit_err + (LINE_CFG.box_penalty_weight * box_pen) + slope_pen

        if score < best_score:
            best_score = score
            best_slope = float(slope)

    return best_slope, float(ax - best_slope * ay)


def compute_column_lines(
    centers: np.ndarray,
    column_ids: np.ndarray,
    row_ids: Optional[np.ndarray] = None,
    boxes: Optional[np.ndarray] = None,
    is_left: bool = True,
) -> List[Tuple[float, float]]:
    if centers is None or len(centers) == 0 or column_ids is None or len(column_ids) == 0:
        return []

    centers = np.asarray(centers, dtype=float)
    column_ids = np.asarray(column_ids, dtype=int)
    row_ids = None if row_ids is None else np.asarray(row_ids, dtype=int)
    boxes = None if boxes is None else np.asarray(boxes, dtype=float)

    uniq = sorted(np.unique(column_ids).tolist())
    lines: List[Tuple[float, float]] = []

    for col_idx, cid in enumerate(uniq):
        mask = column_ids == cid
        pts = centers[mask]
        if len(pts) == 0:
            continue

        anchor_xy = _choose_anchor_point(pts, row_ids, centers, mask)
        max_dev = _column_angle_budget_deg(col_idx, len(uniq), is_left)
        other_boxes = boxes[~mask] if boxes is not None and len(boxes) == len(centers) and np.any(~mask) else None

        slope, intercept = _fit_line_with_anchor_and_constraints(
            pts=pts,
            anchor_xy=anchor_xy,
            other_boxes=other_boxes,
            max_dev_deg=max_dev,
        )
        lines.append((float(slope), float(intercept)))

    if len(lines) <= 1:
        return lines

    sample_y = _sample_y_values(centers, boxes)
    if len(sample_y) == 0:
        return lines

    # Repair: enforce non‑crossing via monotonic projection
    for _ in range(LINE_CFG.repair_iterations):
        repaired_targets: List[np.ndarray] = []
        for yy in sample_y:
            xs = np.array([_line_x_at_y(a, b, float(yy)) for a, b in lines], dtype=float)
            xs_proj = _project_strictly_increasing(xs, LINE_CFG.repair_min_gap_px)
            repaired_targets.append(xs_proj)

        new_lines: List[Tuple[float, float]] = []
        for col_idx, cid in enumerate(uniq):
            mask = column_ids == cid
            pts = centers[mask]
            if len(pts) == 0:
                continue

            anchor_xy = _choose_anchor_point(pts, row_ids, centers, mask)
            max_dev = _column_angle_budget_deg(col_idx, len(uniq), is_left)
            other_boxes = boxes[~mask] if boxes is not None and len(boxes) == len(centers) and np.any(~mask) else None

            target_x = np.array([xs[col_idx] for xs in repaired_targets], dtype=float)
            samples = np.column_stack([target_x, sample_y])
            weights = np.full(len(samples), LINE_CFG.repair_sample_weight, dtype=float)

            slope, intercept = _fit_line_with_anchor_and_constraints(
                pts=pts,
                anchor_xy=anchor_xy,
                other_boxes=other_boxes,
                max_dev_deg=max_dev,
                extra_samples=samples,
                extra_weights=weights,
            )
            new_lines.append((float(slope), float(intercept)))

        if len(new_lines) == len(lines):
            lines = new_lines

    return lines


def validate_geometry(
    centers: np.ndarray,
    boxes: Optional[np.ndarray],
    column_lines: List[Tuple[float, float]],
    column_ids: np.ndarray,
    row_ids: Optional[np.ndarray],
    is_left: bool,
) -> Tuple[bool, List[str]]:
    reasons: List[str] = []

    if centers is None or len(centers) == 0:
        return True, reasons

    centers = np.asarray(centers, dtype=float)
    column_ids = np.asarray(column_ids, dtype=int)
    boxes = None if boxes is None else np.asarray(boxes, dtype=float)
    row_ids = None if row_ids is None else np.asarray(row_ids, dtype=int)

    uniq = sorted(np.unique(column_ids).tolist())
    counts = {int(cid): int(np.sum(column_ids == cid)) for cid in uniq}

    # Capacity constraints
    for cid, cnt in counts.items():
        if cnt > COL_CFG.max_centers_per_column:
            reasons.append(f"column {cid} exceeds max_centers_per_column")
    terminal_cid = uniq[-1] if is_left else uniq[0]
    if counts.get(int(terminal_cid), 0) > COL_CFG.last_column_max_centers:
        reasons.append("terminal column exceeds last_column_max_centers")

    # Angle constraints
    for col_idx, (slope, _) in enumerate(column_lines):
        max_dev = _column_angle_budget_deg(col_idx, len(column_lines), is_left)
        dev = _line_deviation_from_vertical_deg(slope)
        if dev > max_dev + 1e-6:
            reasons.append(f"column {col_idx} angle budget violated")

    # Box collision
    if boxes is not None and len(boxes) == len(centers):
        for col_idx, line in enumerate(column_lines):
            slope, intercept = line
            cid = uniq[col_idx] if col_idx < len(uniq) else col_idx
            other_boxes = boxes[column_ids != cid]
            for box in other_boxes:
                if _box_crosses_line_at_y(slope, intercept, box, LINE_CFG.box_margin_px):
                    reasons.append(f"column {col_idx} crosses a foreign box")
                    break

    # Intersections inside boxes
    if boxes is not None and len(boxes) == len(centers) and len(column_lines) > 1:
        for i in range(len(column_lines)):
            for j in range(i + 1, len(column_lines)):
                inter = _line_intersection_xy(column_lines[i], column_lines[j])
                if inter is None:
                    continue
                x_int, y_int = inter
                if not np.isfinite(x_int) or not np.isfinite(y_int):
                    continue
                for box in boxes:
                    if _point_in_box(x_int, y_int, box, LINE_CFG.box_margin_px):
                        reasons.append(f"intersection of columns {i} and {j} inside a box")
                        break

    # Row ownership: no column should have two points from same row
    if row_ids is not None and len(row_ids) == len(centers):
        for cid in uniq:
            rid_vals = row_ids[column_ids == cid]
            if len(rid_vals) != len(np.unique(rid_vals)):
                reasons.append(f"column {cid} has repeated row ownership")

    return len(reasons) == 0, reasons


# =============================================================================
# Main column solution builder
# =============================================================================

def build_column_solution(
    centers: np.ndarray,
    boxes: Optional[np.ndarray],
    row_ids: Optional[np.ndarray],
    is_left: bool,
) -> Tuple[np.ndarray, List[Tuple[float, float]], float, List[str]]:
    centers = np.asarray(centers, dtype=float)
    row_ids = None if row_ids is None else np.asarray(row_ids, dtype=int)

    best_score = np.inf
    best_column_ids: Optional[np.ndarray] = None
    best_lines: List[Tuple[float, float]] = []
    best_reasons: List[str] = []

    for scale in COL_CFG.recluster_scales[:COL_CFG.max_recluster_rounds]:
        column_ids = _initial_column_ids(
            centers=centers,
            row_ids=row_ids,
            is_left=is_left,
            max_centers_per_column=COL_CFG.max_centers_per_column,
            threshold_scale=float(scale),
        )

        column_ids = split_overfull_columns(centers, column_ids, COL_CFG.max_centers_per_column)
        column_ids = fix_terminal_capacity(
            centers=centers,
            column_ids=column_ids,
            row_ids=row_ids,
            is_left=is_left,
            terminal_cap=COL_CFG.last_column_max_centers,
        )
        column_ids = force_terminal_column_creation(
            centers,
            column_ids,
            row_ids,
            is_left,
        )

        lines = compute_column_lines(
            centers=centers,
            column_ids=column_ids,
            row_ids=row_ids,
            boxes=boxes,
            is_left=is_left,
        )

        valid, reasons = validate_geometry(
            centers=centers,
            boxes=boxes,
            column_lines=lines,
            column_ids=column_ids,
            row_ids=row_ids,
            is_left=is_left,
        )

        uniq = sorted(np.unique(column_ids).tolist()) if len(column_ids) else []
        counts = {int(cid): int(np.sum(column_ids == cid)) for cid in uniq}

        score = 0.0
        score += sum(10.0 for r in reasons if "crosses a box" in r)
        score += sum(12.0 for r in reasons if "inside a box" in r)
        score += sum(4.0 for r in reasons if "angle budget" in r)
        score += sum(3.0 for r in reasons if "exceeds" in r)
        score += sum(2.0 for r in reasons if "repeated row ownership" in r)

        if counts:
            score += float(np.std(list(counts.values())))

        if valid:
            return column_ids, lines, float(score), []

        if score < best_score:
            best_score = score
            best_column_ids = column_ids
            best_lines = lines
            best_reasons = reasons

    if best_column_ids is None:
        best_column_ids = np.zeros(len(centers), dtype=int)
        best_lines = []
        best_reasons = ["failed to build columns"]

    return best_column_ids, best_lines, float(best_score), best_reasons


# =============================================================================
# Main summarization function – public interface (original)
# =============================================================================

def summarize_side_geometry_from_centers(
    centers: np.ndarray,
    boxes: Optional[np.ndarray] = None,
    is_left: bool = True,
    max_rows: int = ROW_CFG.max_rows,
    max_centers_per_column: int = COL_CFG.max_centers_per_column,
) -> dict:
    """
    Compute rows, columns, and lines using the refined algorithm.

    Returns dictionary with keys:
        num_rows, num_cols, row_counts, col_counts, col_center_counts,
        row_lines, column_lines, row_ids, column_ids, num_centers_in_lastrow
    """
    if centers is None or len(centers) == 0:
        return {
            "num_rows": 0,
            "num_cols": 0,
            "row_counts": [],
            "col_counts": {},
            "col_center_counts": {},
            "row_lines": [],
            "column_lines": [],
            "row_ids": np.array([], dtype=int),
            "column_ids": np.array([], dtype=int),
            "num_centers_in_lastrow": 0,
        }

    centers = np.asarray(centers, dtype=float)
    boxes = None if boxes is None else np.asarray(boxes, dtype=float)

    # Rows
    row_lines, row_ids = compute_row_lines(centers, max_rows=max_rows)
    if len(row_lines) == 0:
        row_lines = [(0.0, float(np.mean(centers[:, 1])))]
        row_ids = np.zeros(len(centers), dtype=int)

    row_counts = _counts_from_ids(row_ids, len(row_lines))

    # Columns
    column_ids, column_lines, _score, reasons = build_column_solution(
        centers=centers,
        boxes=boxes,
        row_ids=row_ids,
        is_left=is_left,
    )

    if len(column_ids) == 0:
        column_ids = np.zeros(len(centers), dtype=int)

    # Physical column order (left to right)
    physical_cols = sorted(np.unique(column_ids).tolist())
    raw_col_counts = {int(cid): int(np.sum(column_ids == cid)) for cid in physical_cols}

    # col_center_counts: raw counts per physical column (left‑to‑right)
    col_center_counts = {idx: raw_col_counts[cid] for idx, cid in enumerate(physical_cols)}

    # col_counts: final counts (also raw, because they are already capped)
    col_counts = col_center_counts.copy()

    # bottom row count
    bottom_row_idx = _bottom_row_id(row_ids, centers) if len(row_lines) > 0 else None
    num_centers_in_lastrow = int(np.sum(row_ids == bottom_row_idx)) if bottom_row_idx is not None else 0

    return {
        "num_rows": int(len(row_lines)),
        "num_cols": int(len(physical_cols)),
        "row_counts": [int(x) for x in row_counts],
        "col_counts": col_counts,
        "col_center_counts": col_center_counts,
        "row_lines": row_lines,
        "column_lines": column_lines,
        "row_ids": row_ids,
        "column_ids": column_ids,
        "num_centers_in_lastrow": num_centers_in_lastrow,
    }


# =============================================================================
# NEW: Simple, robust geometry (from debug script) – for production use
# =============================================================================

def fit_line_y_vs_x(points):
    """Fit y = a*x + b, returns (a,b) or None."""
    if len(points) < 2:
        return None
    xs = np.array([p[0] for p in points])
    ys = np.array([p[1] for p in points])
    A = np.vstack([xs, np.ones(len(xs))]).T
    a, b = np.linalg.lstsq(A, ys, rcond=None)[0]
    return a, b


def fit_line_x_vs_y(points):
    """Fit x = a*y + b, returns (a,b) or None."""
    if len(points) < 2:
        return None
    ys = np.array([p[1] for p in points])
    xs = np.array([p[0] for p in points])
    A = np.vstack([ys, np.ones(len(ys))]).T
    a, b = np.linalg.lstsq(A, xs, rcond=None)[0]
    return a, b


def cluster_rows_simple(centers, boxes):
    """
    Cluster points into rows (1, 2, or 3 rows) using Y-gaps and box heights.
    """
    centers = np.asarray(centers)
    if len(centers) == 0:
        return []
    if len(centers) == 1:
        return [[0]]

    sorted_idx = np.argsort(centers[:, 1])
    sorted_centers = centers[sorted_idx]
    gaps = sorted_centers[1:, 1] - sorted_centers[:-1, 1]

    if len(gaps) == 1:
        heights = boxes[:, 3] - boxes[:, 1] if len(boxes) > 0 else [50]
        avg_h = np.median(heights)
        if gaps[0] > avg_h * 0.8:
            return [sorted_idx[:1].tolist(), sorted_idx[1:].tolist()]
        else:
            return [sorted_idx.tolist()]

    gap_indices = np.argsort(gaps)[::-1]
    g1_idx = gap_indices[0]
    g1 = gaps[g1_idx]
    g2 = gaps[gap_indices[1]] if len(gaps) > 1 else 0

    if g1 > 1.8 * g2 and g2 > 0:
        split_indices = [g1_idx]
    else:
        split_indices = sorted([g1_idx, gap_indices[1]]) if len(gaps) >= 2 else [g1_idx]

    rows = []
    start = 0
    for split in split_indices:
        rows.append(sorted_idx[start:split+1])
        start = split+1
    rows.append(sorted_idx[start:])

    # Cap at 3 rows by merging closest
    while len(rows) > 3:
        min_gap = float('inf')
        merge_idx = 0
        for i in range(len(rows)-1):
            y1 = np.mean(centers[rows[i], 1])
            y2 = np.mean(centers[rows[i+1], 1])
            gap = abs(y2 - y1)
            if gap < min_gap:
                min_gap = gap
                merge_idx = i
        merged = np.concatenate([rows[merge_idx], rows[merge_idx+1]])
        rows[merge_idx] = merged.tolist()
        del rows[merge_idx+1]

    return rows


def summarize_side_geometry_simple(centers, boxes, is_left=True):
    """
    Lightweight, robust side geometry using dynamic row clustering and
    parallel line fitting. Returns dict with:
        num_rows, num_cols, row_counts, col_counts,
        col_center_counts, row_lines, column_lines.
    """
    centers = np.asarray(centers) if centers is not None else np.empty((0, 2))
    boxes = np.asarray(boxes) if boxes is not None else np.empty((0, 4))

    if len(centers) == 0:
        return {
            "num_rows": 0, "num_cols": 0,
            "row_counts": [], "col_counts": {},
            "col_center_counts": {},
            "row_lines": [], "column_lines": []
        }

    # ---- 1. Rows ----
    rows = cluster_rows_simple(centers, boxes)
    row_avg_y = [np.mean(centers[r, 1]) for r in rows]
    order = np.argsort(row_avg_y)
    rows = [rows[i] for i in order]

    row_bboxes = []
    for r in rows:
        r_sorted = sorted(r, key=lambda idx: centers[idx, 0])
        row_bboxes.append(r_sorted)

    row_counts = [len(r) for r in row_bboxes]
    num_rows = len(row_counts)

    # ---- Fit row lines with forced parallel slope ----
    row_slopes = []
    for r in row_bboxes:
        if len(r) > 0:
            pts = [(centers[idx, 0], centers[idx, 1]) for idx in r]
            ab = fit_line_y_vs_x(pts)
            if ab is not None:
                row_slopes.append(ab[0])
            else:
                row_slopes.append(0.0)
        else:
            row_slopes.append(0.0)

    avg_row_slope = np.median(row_slopes) if row_slopes else 0.0
    row_lines = []
    for r in row_bboxes:
        if len(r) > 0:
            avg_x = np.mean([centers[idx, 0] for idx in r])
            avg_y = np.mean([centers[idx, 1] for idx in r])
            b = avg_y - avg_row_slope * avg_x
            row_lines.append((avg_row_slope, b))
        else:
            row_lines.append((0.0, 0.0))

    if num_rows == 0:
        return {
            "num_rows": 0, "num_cols": 0,
            "row_counts": [], "col_counts": {},
            "col_center_counts": {},
            "row_lines": [], "column_lines": []
        }

    # ---- 2. Reference row (most boxes) ----
    ref_row_idx = max(range(num_rows), key=lambda i: row_counts[i])
    ref_indices = row_bboxes[ref_row_idx]
    if len(ref_indices) == 0:
        return {
            "num_rows": num_rows, "num_cols": 0,
            "row_counts": row_counts, "col_counts": {},
            "col_center_counts": {},
            "row_lines": row_lines, "column_lines": []
        }

    anchor_x = [centers[idx, 0] for idx in ref_indices]
    if len(anchor_x) > 1:
        diffs = [anchor_x[i+1] - anchor_x[i] for i in range(len(anchor_x)-1)]
        avg_spacing = np.median(diffs)
    else:
        avg_spacing = 50.0

    match_threshold = avg_spacing * 0.6

    # ---- 3. Match anchors to other rows ----
    matches_per_anchor = [[] for _ in range(len(anchor_x))]
    for row_i, row_indices in enumerate(row_bboxes):
        if row_i == ref_row_idx:
            continue
        for anchor_pos, x_anchor in enumerate(anchor_x):
            best_idx = None
            best_dist = float('inf')
            for idx in row_indices:
                dist = abs(centers[idx, 0] - x_anchor)
                if dist < best_dist:
                    best_dist = dist
                    best_idx = idx
            if best_idx is not None and best_dist < match_threshold:
                matches_per_anchor[anchor_pos].append((row_i, best_idx))

    # ---- 4. Build column candidates ----
    column_candidates = []  # (x, rows_set, indices)
    for anchor_pos, x_anchor in enumerate(anchor_x):
        rows_in_col = {ref_row_idx}
        indices_in_col = [ref_indices[anchor_pos]]
        for (row_i, idx) in matches_per_anchor[anchor_pos]:
            rows_in_col.add(row_i)
            indices_in_col.append(idx)
        column_candidates.append((x_anchor, rows_in_col, indices_in_col))

    # ---- 5. Add unmatched boxes from other rows ----
    covered_indices = set(ref_indices)
    for matches in matches_per_anchor:
        for (_, idx) in matches:
            covered_indices.add(idx)

    for row_i, row_indices in enumerate(row_bboxes):
        if row_i == ref_row_idx:
            continue
        for idx in row_indices:
            if idx not in covered_indices:
                x = centers[idx, 0]
                column_candidates.append((x, {row_i}, [idx]))

    # ---- 6. Merge close candidates ----
    column_candidates.sort(key=lambda p: p[0])
    merge_threshold = 0.6 * avg_spacing

    clusters = []
    if column_candidates:
        current = [column_candidates[0]]
        for i in range(1, len(column_candidates)):
            if column_candidates[i][0] - column_candidates[i-1][0] < merge_threshold:
                current.append(column_candidates[i])
            else:
                clusters.append(current)
                current = [column_candidates[i]]
        clusters.append(current)

    # ---- 7. Build column lines with forced parallel slope ----
    col_slopes = []
    cluster_centroids = []
    for cluster in clusters:
        all_indices = []
        for (_, _, indices) in cluster:
            all_indices.extend(indices)
        pts = [(centers[idx, 0], centers[idx, 1]) for idx in all_indices]
        ab = fit_line_x_vs_y(pts)
        if ab is not None:
            col_slopes.append(ab[0])
        else:
            col_slopes.append(0.0)
        avg_x = np.mean([p[0] for p in pts])
        avg_y = np.mean([p[1] for p in pts])
        cluster_centroids.append((avg_x, avg_y))

    avg_col_slope = np.median(col_slopes) if col_slopes else 0.0

    column_lines = []
    col_counts = {}
    col_center_counts = {}

    for col_idx, (cluster, (avg_x, avg_y)) in enumerate(zip(clusters, cluster_centroids)):
        b = avg_x - avg_col_slope * avg_y
        column_lines.append((avg_col_slope, b))

        all_rows = set()
        for (_, rows_set, _) in cluster:
            all_rows.update(rows_set)
        count = len(all_rows)
        col_counts[col_idx] = count
        col_center_counts[col_idx] = count

    num_cols = len(column_lines)

    return {
        "num_rows": num_rows,
        "num_cols": num_cols,
        "row_counts": row_counts,
        "col_counts": col_counts,
        "col_center_counts": col_center_counts,
        "row_lines": row_lines,
        "column_lines": column_lines
    }