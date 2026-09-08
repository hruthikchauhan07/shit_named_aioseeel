"""
Top‑view cylinder classification.
Detects:
- 425‑kg (JUMBO) via bounding‑box area.
- 14.5‑kg (STRAIGHT) and 19‑kg (ZIG‑ZAG) via column‑wise offset analysis.
- MIXED only if a significant portion of columns are ZIG‑ZAG.
"""

import logging
import numpy as np
from collections import Counter, defaultdict
from typing import Iterable, Optional
from sklearn.cluster import DBSCAN
from sklearn.linear_model import LinearRegression
from scipy.optimize import linear_sum_assignment
from scipy.spatial import Delaunay

from utils import cluster_1d, rotate_points

logger = logging.getLogger(__name__)

# =========================
# CONFIG
# =========================
LARGE_BBOX_AREA_THRESHOLD = 35000   # increased to avoid false JUMBO from blurry images

# Column‑wise detection – tightened to avoid false ZIG‑ZAG
ROW_EPS = 35
COL_SPACING_THRESH = 0.25          # below this → STRAIGHT
ZIGZAG_THRESH_LOW = 0.40
ZIGZAG_THRESH_HIGH = 0.75

# Tolerance: ignore few ZIG‑ZAG columns
MIN_ZIGZAG_COLUMNS_TO_BE_MIXED = 2
ZIGZAG_RATIO_THRESH = 0.1

# =========================
# 425 DETECTION
# =========================
def split_large_and_small(detections):
    if not detections:
        return (np.array([], dtype=bool), np.empty((0,1), dtype=np.float32), np.array([], dtype=int))
    areas = np.array([float(d["bbox_area"]) for d in detections], dtype=np.float32)
    large_mask = areas >= LARGE_BBOX_AREA_THRESHOLD
    large_count = int(np.sum(large_mask))
    # 🆕 If fewer than 3 large detections, treat them as normal (avoid false JUMBO)
    if large_count > 0 and large_count < 3:
        logger.info(f"[425 SPLIT] Found {large_count} large detections, but less than 3 – ignoring JUMBO flag")
        large_mask = np.zeros_like(large_mask, dtype=bool)
        large_count = 0
    logger.info("[425 SPLIT] total=%d 425=%d other=%d", len(areas), large_count, int(np.sum(~large_mask)))
    return large_mask, areas.reshape(-1,1), np.zeros(len(areas), dtype=int)

# =========================
# ALIGNMENT
# =========================
def estimate_angle(points):
    points = np.asarray(points, dtype=np.float32)
    if len(points) < 2:
        return 0.0
    center = np.mean(points, axis=0)
    shifted = points - center
    covariance = np.cov(shifted.T)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    principal_axis = eigenvectors[:, np.argmax(eigenvalues)]
    angle = np.degrees(np.arctan2(principal_axis[1], principal_axis[0]))
    return angle

def rotate_points_around_centroid(points, angle):
    points = np.asarray(points, dtype=np.float32)
    center = np.mean(points, axis=0)
    shifted = points - center
    theta = np.radians(angle)
    R = np.array([[np.cos(theta), -np.sin(theta)],
                  [np.sin(theta), np.cos(theta)]], dtype=np.float32)
    rotated = shifted @ R.T
    return rotated, center, R

def align_points(points):
    angle = estimate_angle(points)
    rotated, center, R = rotate_points_around_centroid(points, -angle)
    return rotated, center, R, angle

# =========================
# ROW/COLUMN DETECTION
# =========================
def cluster_axis(values, eps=35):
    values = np.asarray(values, dtype=np.float32).reshape(-1,1)
    db = DBSCAN(eps=eps, min_samples=1)
    return db.fit_predict(values)

def find_rows(aligned_points, row_eps=ROW_EPS):
    aligned_points = np.asarray(aligned_points, dtype=np.float32)
    ys = aligned_points[:, 1]
    row_labels = cluster_axis(ys, eps=row_eps)
    unique_labels = np.unique(row_labels)
    ordered_labels = sorted(unique_labels, key=lambda label: np.mean(ys[row_labels == label]))
    rows = []
    for label in ordered_labels:
        indices = np.where(row_labels == label)[0]
        row = []
        for idx in indices:
            row.append({
                "index": idx,
                "point": aligned_points[idx],
                "x": float(aligned_points[idx][0]),
                "y": float(aligned_points[idx][1])
            })
        row.sort(key=lambda p: p["x"])
        rows.append(row)
    return rows, row_labels

def assign_columns(rows):
    if len(rows) == 0:
        return rows
    rows.sort(key=lambda r: np.mean([p["y"] for p in r]))
    for row in rows:
        row.sort(key=lambda p: p["x"])
    for c, pt in enumerate(rows[0], start=1):
        pt["column"] = c
    for r in range(1, len(rows)):
        prev = sorted(rows[r-1], key=lambda p: p["column"])
        curr = rows[r]
        shift = np.median([
            c["x"] - p["x"]
            for p, c in zip(prev[:min(len(prev), len(curr))],
                            curr[:min(len(prev), len(curr))])
        ])
        cost = np.zeros((len(curr), len(prev)))
        for i, cp in enumerate(curr):
            for j, pp in enumerate(prev):
                expected_x = pp["x"] + shift
                cost[i,j] = abs(cp["x"] - expected_x)
        row_ind, col_ind = linear_sum_assignment(cost)
        for i,j in zip(row_ind, col_ind):
            curr[i]["column"] = prev[j]["column"]
        next_col = max(p["column"] for p in prev) + 1
        for pt in curr:
            if "column" not in pt:
                pt["column"] = next_col
                next_col += 1
    return rows

# =========================
# PATTERN DETECTION (with tolerance)
# =========================
def detect_pattern_by_columns(rows):
    if len(rows) < 2:
        return "STRAIGHT", None

    col_data = defaultdict(list)
    for row_idx, row in enumerate(rows):
        for p in row:
            col_data[p["column"]].append((row_idx, p["x"]))

    if len(col_data) < 2:
        return "STRAIGHT", None

    first_row = rows[0]
    if len(first_row) < 2:
        return "STRAIGHT", None
    x_vals = [p["x"] for p in first_row]
    x_vals.sort()
    gaps = [x_vals[i+1] - x_vals[i] for i in range(len(x_vals)-1)]
    if not gaps:
        return "STRAIGHT", None
    col_spacing = np.median(gaps)

    col_patterns = {}
    for col, positions in col_data.items():
        positions.sort()
        if len(positions) < 2:
            continue
        offsets = []
        for i in range(len(positions)-1):
            row1, x1 = positions[i]
            row2, x2 = positions[i+1]
            if row2 - row1 == 1:
                offsets.append(x2 - x1)
        if offsets:
            median_offset = np.median(offsets)
            norm_offset = median_offset / col_spacing
            if abs(norm_offset) < COL_SPACING_THRESH:
                col_patterns[col] = "STRAIGHT"
            elif ZIGZAG_THRESH_LOW < abs(norm_offset) < ZIGZAG_THRESH_HIGH:
                col_patterns[col] = "ZIG-ZAG"
            else:
                col_patterns[col] = "UNDEFINED"

    if not col_patterns:
        return "STRAIGHT", None

    straight_cols = [c for c,p in col_patterns.items() if p == "STRAIGHT"]
    zigzag_cols = [c for c,p in col_patterns.items() if p == "ZIG-ZAG"]

    if not zigzag_cols:
        return "STRAIGHT", None

    total_cols = len(col_patterns)
    if len(zigzag_cols) < MIN_ZIGZAG_COLUMNS_TO_BE_MIXED:
        logger.debug(f"Only {len(zigzag_cols)} ZIG-ZAG columns, treating as STRAIGHT")
        return "STRAIGHT", None

    if len(zigzag_cols) / total_cols < ZIGZAG_RATIO_THRESH:
        logger.debug(f"ZIG-ZAG ratio {len(zigzag_cols)/total_cols:.2f} < {ZIGZAG_RATIO_THRESH}, treating as STRAIGHT")
        return "STRAIGHT", None

    # Now we have both patterns -> MIXED
    all_cols = sorted(col_patterns.keys())
    for i in range(1, len(all_cols)):
        if col_patterns[all_cols[i-1]] != col_patterns[all_cols[i]]:
            split_col = all_cols[i]
            return "MIXED", split_col

    return "STRAIGHT", None

# =========================
# MAIN CLASSIFICATION
# =========================
def _pattern_summary_from_centers(centers: np.ndarray, use_leftmost: bool) -> dict:
    if centers is None or len(centers) == 0:
        return {
            "pattern_type": "STRAIGHT",
            "count": 0,
            "num_rows": 0,
            "num_cols": 0,
            "row_counts": [],
            "kg14_count": 0,
            "kg19_count": 0,
            "truck_type": "EMPTY",
            "cylinder_color": None,
            "regions": [],
        }

    aligned, _, _, _ = align_points(centers)
    rows, _ = find_rows(aligned, row_eps=ROW_EPS)
    if len(rows) == 0:
        summary = summarize_geometry_from_centers(centers, use_leftmost, zigzag_flag=False)
        return {
            "pattern_type": "STRAIGHT",
            "count": summary["count"],
            "num_rows": summary["num_rows"],
            "num_cols": summary["num_cols"],
            "row_counts": summary["row_counts"],
            "kg14_count": summary["count"],
            "kg19_count": 0,
            "truck_type": "14.5 KG TRUCK",
            "cylinder_color": "RED",
            "regions": [],
        }

    rows = assign_columns(rows)
    pattern_type, split_col = detect_pattern_by_columns(rows)

    num_rows = min(len(rows), 7)
    num_cols = max([len(r) for r in rows]) if rows else 0
    num_cols = min(num_cols, 19)

    if pattern_type == "STRAIGHT":
        kg14_count = len(centers)
        kg19_count = 0
        truck_type = "14.5 KG TRUCK"
        cylinder_color = "RED"
        regions = []
    elif pattern_type == "ZIG-ZAG":
        kg14_count = 0
        kg19_count = len(centers)
        truck_type = "19 KG TRUCK"
        cylinder_color = "BLUE"
        regions = []
    else:  # MIXED
        if split_col is not None:
            kg14_total = 0
            kg19_total = 0
            resolved_regions = []
            min_x_all = np.min(aligned[:, 0])
            max_x_all = np.max(aligned[:, 0])

            for region in [
                {"label": "STRAIGHT", "start_x": min_x_all, "end_x": split_col},
                {"label": "ZIG-ZAG", "start_x": split_col, "end_x": max_x_all}
            ]:
                label = region["label"]
                start_x = region["start_x"]
                end_x = region["end_x"]
                mask = (aligned[:, 0] >= start_x) & (aligned[:, 0] < end_x)
                region_pts_aligned = aligned[mask]
                if len(region_pts_aligned) == 0:
                    resolved_regions.append({
                        "label": label,
                        "start_x": int(start_x),
                        "end_x": int(end_x),
                        "count": 0,
                        "num_cols": 0,
                        "num_rows": 0,
                        "row_counts": []
                    })
                    continue

                region_centers = centers[mask]
                region_summary = summarize_geometry_from_centers(
                    region_centers,
                    use_leftmost=use_leftmost,
                    zigzag_flag=(label == "ZIG-ZAG")
                )
                region_count = region_summary["count"]
                if label == "STRAIGHT":
                    kg14_total += region_count
                else:
                    kg19_total += region_count

                resolved_regions.append({
                    "label": label,
                    "start_x": int(start_x),
                    "end_x": int(end_x),
                    "count": region_count,
                    "num_cols": region_summary["num_cols"],
                    "num_rows": region_summary["num_rows"],
                    "row_counts": region_summary["row_counts"],
                    "top_count": region_count,
                    "fixed_rows": 2 if label == "ZIG-ZAG" else None,
                })

            kg14_count = kg14_total
            kg19_count = kg19_total
            truck_type = "MIXED LOAD"
            cylinder_color = None
            regions = resolved_regions
        else:
            kg14_count = len(centers)
            kg19_count = 0
            truck_type = "14.5 KG TRUCK"
            cylinder_color = "RED"
            regions = []

    return {
        "pattern_type": pattern_type,
        "count": kg14_count + kg19_count,
        "num_rows": num_rows,
        "num_cols": num_cols,
        "row_counts": [len(r) for r in rows[:num_rows]],
        "kg14_count": kg14_count,
        "kg19_count": kg19_count,
        "truck_type": truck_type,
        "cylinder_color": cylinder_color,
        "regions": regions,
    }

# =========================
# EXISTING HELPER FUNCTIONS (unchanged)
# =========================
def compute_top_count_from_centers(
    centers: np.ndarray,
    use_leftmost: bool,
    zigzag_flag: bool = False,
) -> int:
    """Original function – kept for backward compatibility."""
    if centers is None or len(centers) == 0:
        return 0
    if len(centers) < 5:
        return int(len(centers))
    centers = np.asarray(centers, dtype=float)
    X = centers[:, 0].reshape(-1, 1)
    Y = centers[:, 1]
    slope = LinearRegression().fit(X, Y).coef_[0]
    rotated = rotate_points(centers, -np.arctan(slope))
    row_clusters = cluster_1d(rotated[:, 1].tolist(), tol=40)
    row_lines, rows_points = [], []
    for rc in row_clusters:
        mask = np.isin(rotated[:, 1], rc)
        pts = centers[mask]
        if len(pts) < 2:
            continue
        rows_points.append(pts)
        reg = LinearRegression().fit(pts[:, 0].reshape(-1, 1), pts[:, 1])
        row_lines.append((reg.coef_[0], reg.intercept_))
    if not row_lines:
        return int(len(centers))
    row_counts = [0] * len(row_lines)
    for x, y in centers:
        dists = [abs(a*x - y + b)/np.sqrt(a*a+1) for a, b in row_lines]
        row_counts[np.argmin(dists)] += 1
    mode_row_count = Counter(row_counts).most_common(1)[0][0]
    col_pts = []
    for row in rows_points:
        sorted_row = row[row[:, 0].argsort()]
        col_pts.append(sorted_row[0] if use_leftmost else sorted_row[-1])
    col_pts = np.array(col_pts)
    if len(col_pts) < 2:
        return int(len(centers))
    reg = LinearRegression().fit(col_pts[:, 1].reshape(-1, 1), col_pts[:, 0])
    gamma, delta = reg.coef_[0], reg.intercept_
    tolerance = 15
    col_count = sum(1 for x, y in centers if abs(x - (gamma*y + delta)) < tolerance)
    if col_count >= 8:
        col_count = 7
    elif col_count == 6:
        col_count = 7
    if zigzag_flag:
        col_groups = cluster_1d(centers[:, 0].tolist(), tol=12)
        col_groups = [g for g in col_groups if len(g) >= 2]
        num_cols = len(col_groups)
        if num_cols == 0:
            return int(len(centers))
        col_counts = [len(g) for g in col_groups]
        mode_col = Counter(col_counts).most_common(1)[0][0]
        HEIGHT = 2
        return int(num_cols * mode_col * HEIGHT)
    return int(mode_row_count * col_count)

def summarize_geometry_from_centers(
    centers: np.ndarray,
    use_leftmost: bool,
    zigzag_flag: bool = False,
) -> dict:
    """Original function – kept for backward compatibility."""
    if centers is None or len(centers) == 0:
        return {"count": 0, "num_rows": 0, "num_cols": 0, "row_counts": []}
    if len(centers) < 5:
        return {
            "count": int(len(centers)),
            "num_rows": 1,
            "num_cols": int(len(centers)),
            "row_counts": [int(len(centers))],
        }
    centers = np.asarray(centers, dtype=float)
    X = centers[:, 0].reshape(-1, 1)
    Y = centers[:, 1]
    slope = LinearRegression().fit(X, Y).coef_[0]
    rotated = rotate_points(centers, -np.arctan(slope))
    row_clusters = cluster_1d(rotated[:, 1].tolist(), tol=40)
    row_lines, rows_points = [], []
    for rc in row_clusters:
        mask = np.isin(rotated[:, 1], rc)
        pts = centers[mask]
        if len(pts) < 2:
            continue
        rows_points.append(pts)
        reg = LinearRegression().fit(pts[:, 0].reshape(-1, 1), pts[:, 1])
        row_lines.append((reg.coef_[0], reg.intercept_))
    if not row_lines:
        return {"count": len(centers), "num_rows": 1, "num_cols": len(centers), "row_counts": [len(centers)]}
    row_counts = [0] * len(row_lines)
    for x, y in centers:
        dists = [abs(a*x - y + b)/np.sqrt(a*a+1) for a, b in row_lines]
        row_counts[np.argmin(dists)] += 1
    mode_row_count = Counter(row_counts).most_common(1)[0][0]
    num_rows = len(row_lines)
    if num_rows > 7:
        logger.warning(f"[GEOMETRY] Detected {num_rows} rows, capping to 7")
        num_rows = 7
    col_pts = []
    for row in rows_points:
        sorted_row = row[row[:, 0].argsort()]
        col_pts.append(sorted_row[0] if use_leftmost else sorted_row[-1])
    col_pts = np.array(col_pts)
    if len(col_pts) < 2:
        return {"count": len(centers), "num_rows": num_rows, "num_cols": mode_row_count, "row_counts": row_counts}
    reg = LinearRegression().fit(col_pts[:, 1].reshape(-1, 1), col_pts[:, 0])
    gamma, delta = reg.coef_[0], reg.intercept_
    tolerance = 15
    col_count = sum(1 for x, y in centers if abs(x - (gamma*y + delta)) < tolerance)
    if col_count >= 8:
        col_count = 7
    elif col_count == 6:
        col_count = 7
    if zigzag_flag:
        col_groups = cluster_1d(centers[:, 0].tolist(), tol=12)
        col_groups = [g for g in col_groups if len(g) >= 2]
        num_cols = len(col_groups)
        if num_cols == 0:
            num_cols = col_count
        count = int(num_cols * mode_row_count)
    else:
        num_cols = col_count
        count = int(mode_row_count * num_cols)
    return {
        "count": count,
        "num_rows": num_rows,
        "num_cols": mode_row_count,
        "row_counts": row_counts,
    }

def summarize_side_geometry_from_centers(
    centers: np.ndarray,
) -> dict:
    """Original side‑view summary – kept unchanged."""
    if centers is None or len(centers) == 0:
        return {"num_rows": 0, "num_cols": 0, "col_counts": {}}
    centers = np.asarray(centers, dtype=float)
    if len(centers) < 3:
        return {
            "num_rows": int(len(centers)),
            "num_cols": 1,
            "col_counts": {0: int(len(centers))},
        }
    col_groups = cluster_1d(centers[:, 0].tolist(), tol=20)
    col_groups_sorted = sorted(col_groups, key=lambda g: np.mean(g))
    col_counts = {}
    row_count_candidates = []
    for idx, group in enumerate(col_groups_sorted):
        mask = np.isin(centers[:, 0], group)
        pts = centers[mask]
        col_counts[idx] = int(len(pts))
        row_count_candidates.append(len(pts))
    num_cols = len(col_counts)
    if row_count_candidates:
        num_rows = int(Counter(row_count_candidates).most_common(1)[0][0])
    else:
        num_rows = 0
    return {"num_rows": num_rows, "num_cols": num_cols, "col_counts": col_counts}

def summarize_side_regions_from_centers(
    centers: np.ndarray,
    regions: list,
) -> list:
    """Original side regions – kept unchanged."""
    if centers is None or len(centers) == 0 or not regions:
        return []
    centers = np.asarray(centers, dtype=float)
    centers = centers[np.argsort(centers[:, 0])];
    resolved = []
    for region in regions:
        label = region["label"]
        start_x = region["start_x"]
        end_x = region["end_x"]
        mask = (centers[:, 0] >= start_x) & (centers[:, 0] < end_x)
        region_pts = centers[mask]
        if len(region_pts) == 0:
            summary = {"num_rows": 0, "num_cols": 0, "col_counts": {}}
        else:
            summary = summarize_side_geometry_from_centers(region_pts)
        resolved.append({
            "label": label,
            "start_x": int(start_x),
            "end_x": int(end_x),
            "num_rows": summary["num_rows"],
            "num_cols": summary["num_cols"],
            "col_counts": summary["col_counts"],
        })
    return resolved

# =========================
# MAIN API
# =========================
def classify_top_load(detections: Iterable[dict], use_leftmost: bool = True) -> dict:
    """
    Unified top-load classifier.
    """
    detections = list(detections or [])
    if not detections:
        return {
            "kind": "EMPTY",
            "pattern_type": "STRAIGHT",
            "truck_type": "EMPTY",
            "count": 0,
            "num_rows": 0,
            "num_cols": 0,
            "row_counts": [],
            "kg425_count": 0,
            "kg14_count": 0,
            "kg19_count": 0,
            "cylinder_color": None,
            "is_jumbo": False,
            "regions": [],
        }

    large_mask, _, _ = split_large_and_small(detections)
    large_count = int(np.sum(large_mask))
    total_count = len(detections)

    if large_count == total_count:
        return {
            "kind": "JUMBO",
            "pattern_type": "JUMBO",
            "truck_type": "425 KG JUMBO TRUCK",
            "count": total_count,
            "num_rows": 0,
            "num_cols": 0,
            "row_counts": [],
            "kg425_count": total_count,
            "kg14_count": 0,
            "kg19_count": 0,
            "cylinder_color": None,
            "is_jumbo": True,
            "regions": [],
        }

    centers = np.array([[d["cx"], d["cy"]] for d in detections], dtype=np.float32)
    other_centers = centers[~large_mask] if large_count > 0 else centers
    secondary = _pattern_summary_from_centers(other_centers, use_leftmost)

    if large_count > 0:
        return {
            "kind": "JUMBO_MIXED",
            "pattern_type": secondary["pattern_type"],
            "truck_type": f"425 KG + {secondary['truck_type']}",
            "count": large_count + secondary["count"],
            "num_rows": secondary["num_rows"],
            "num_cols": secondary["num_cols"],
            "row_counts": secondary["row_counts"],
            "kg425_count": large_count,
            "kg14_count": secondary["kg14_count"],
            "kg19_count": secondary["kg19_count"],
            "cylinder_color": secondary["cylinder_color"],
            "is_jumbo": True,
            "regions": secondary.get("regions", []),
        }
    else:
        return {
            "kind": "NORMAL",
            "pattern_type": secondary["pattern_type"],
            "truck_type": secondary["truck_type"],
            "count": secondary["count"],
            "num_rows": secondary["num_rows"],
            "num_cols": secondary["num_cols"],
            "row_counts": secondary["row_counts"],
            "kg425_count": 0,
            "kg14_count": secondary["kg14_count"],
            "kg19_count": secondary["kg19_count"],
            "cylinder_color": secondary["cylinder_color"],
            "is_jumbo": False,
            "regions": secondary.get("regions", []),
        }