#!/usr/bin/env python3
"""
Side-view debug script – clean, deterministic row/column detection.
Rows: 1–3 rows based on y‑gap ratio.
Columns: anchor on the row with most boxes, add unmatched boxes, merge close ones.
Lines: straight horizontal/vertical (no perspective fitting).
"""

import os
import sys
import cv2
import numpy as np
from ultralytics import YOLO
import logging

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ================================================================
# 📌 EDIT THESE PATHS
# ================================================================
SIDE1_PATH = r"C:\Users\admin\Desktop\cyl\approacg-script-backup\side_images\TN69BZ7291_cam4_20260723_181133_777230_early.jpg"
SIDE2_PATH = r"C:\Users\admin\Desktop\cyl\approacg-script-backup\side_images\TN69BZ7291_cam1_20260724_165756_189923_early.jpg"
SIDE_MODEL_PATH = r"C:\Users\admin\Desktop\cyl\approacg-script-backup\models\side_v8m.pt"

CONF_THRESHOLD = 0.15

# ================================================================
# Helper functions
# ================================================================

def letterbox(img, new_shape=(640, 640), color=(114, 114, 114)):
    shape = img.shape[:2]
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
    dw, dh = dw // 2, dh // 2
    if shape[::-1] != new_unpad:
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = dh, dh
    left, right = dw, dw
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return img, dw, dh, r


def scale_boxes_to_original(boxes_640, dw, dh, r, orig_w, orig_h):
    if len(boxes_640) == 0:
        return np.empty((0, 4), dtype=float)
    boxes = boxes_640.astype(float).copy()
    boxes[:, 0] = (boxes[:, 0] - dw) / r
    boxes[:, 2] = (boxes[:, 2] - dw) / r
    boxes[:, 1] = (boxes[:, 1] - dh) / r
    boxes[:, 3] = (boxes[:, 3] - dh) / r
    boxes[:, 0] = np.clip(boxes[:, 0], 0, orig_w - 1)
    boxes[:, 2] = np.clip(boxes[:, 2], 0, orig_w - 1)
    boxes[:, 1] = np.clip(boxes[:, 1], 0, orig_h - 1)
    boxes[:, 3] = np.clip(boxes[:, 3], 0, orig_h - 1)
    return boxes


def boxes_to_centers(boxes: np.ndarray) -> np.ndarray:
    if boxes is None or len(boxes) == 0:
        return np.empty((0, 2), dtype=float)
    return np.stack([(boxes[:, 0] + boxes[:, 2]) / 2.0,
                     (boxes[:, 1] + boxes[:, 3]) / 2.0], axis=1)


def non_max_suppression(boxes, iou_threshold=0.5):
    if len(boxes) == 0:
        return boxes
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = np.argsort(areas)[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0, xx2 - xx1)
        h = np.maximum(0, yy2 - yy1)
        inter = w * h
        iou = inter / (areas[i] + areas[order[1:]] - inter)
        order = order[1:][iou < iou_threshold]
    return boxes[keep]


def infer_side(frame, model):
    orig_h, orig_w = frame.shape[:2]
    infer_img, dw, dh, r = letterbox(frame, (640, 640))
    results = model(infer_img, conf=CONF_THRESHOLD, verbose=False)[0]
    boxes_640 = results.boxes.xyxy.cpu().numpy() if results.boxes is not None else np.empty((0, 4), dtype=float)
    if len(boxes_640) == 0:
        return None, None
    boxes = scale_boxes_to_original(boxes_640, dw, dh, r, orig_w, orig_h)
    boxes = non_max_suppression(boxes, iou_threshold=0.5)
    centers = boxes_to_centers(boxes)
    return boxes, centers


# ================================================================
# Robust row clustering – 1, 2, or 3 rows
# ================================================================

def cluster_rows(centers, boxes):
    if len(centers) == 0:
        return []
    if len(centers) == 1:
        return [[0]]

    sorted_idx = np.argsort(centers[:, 1])
    sorted_centers = centers[sorted_idx]
    gaps = sorted_centers[1:, 1] - sorted_centers[:-1, 1]

    if len(gaps) == 1:
        heights = boxes[:, 3] - boxes[:, 1]
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


# ================================================================
# Geometry computation – straight horizontal/vertical lines
# ================================================================

def compute_geometry(centers, boxes, is_left=True):
    if len(centers) == 0:
        return {
            "num_rows": 0, "num_cols": 0,
            "row_counts": [], "col_counts": {},
            "col_center_counts": {},
            "row_lines": [], "column_lines": []
        }

    # ---- 1. Rows ----
    rows = cluster_rows(centers, boxes)
    row_avg_y = [np.mean(centers[r, 1]) for r in rows]
    order = np.argsort(row_avg_y)
    rows = [rows[i] for i in order]

    row_bboxes = []
    for r in rows:
        r_sorted = sorted(r, key=lambda idx: centers[idx, 0])
        row_bboxes.append(r_sorted)

    row_counts = [len(r) for r in row_bboxes]
    num_rows = len(row_counts)

    # Row lines: horizontal at average y
    row_lines = []
    for r in row_bboxes:
        if len(r) > 0:
            avg_y = np.mean([centers[idx, 1] for idx in r])
            row_lines.append((0.0, avg_y))   # y = constant
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

    # ---- 4. Build column candidates from anchors (keep all) ----
    column_candidates = []   # (x, rows_set)
    for anchor_pos, x_anchor in enumerate(anchor_x):
        rows_in_col = {ref_row_idx}
        for (row_i, _) in matches_per_anchor[anchor_pos]:
            rows_in_col.add(row_i)
        column_candidates.append((x_anchor, rows_in_col))

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
                column_candidates.append((x, {row_i}))

    # ---- 6. Merge close candidates ----
    column_candidates.sort(key=lambda p: p[0])
    # Use the reference row's avg_spacing for merge threshold (more stable)
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

    # ---- 7. Build column lines (vertical at average x) ----
    column_lines = []
    col_counts = {}
    col_center_counts = {}

    for col_idx, cluster in enumerate(clusters):
        avg_x = np.mean([p[0] for p in cluster])
        column_lines.append((0.0, avg_x))   # x = constant

        all_rows = set()
        for (_, rows_set) in cluster:
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
        "row_lines": row_lines,          # (a,b) for y = a*x+b with a=0
        "column_lines": column_lines     # (a,b) for x = a*y+b with a=0
    }


# ================================================================
# Annotation – bold boxes, straight lines, highlighted centroids
# ================================================================

BOX_COLOR = (128, 0, 0)
BOX_THICKNESS = 3
CENTER_COLOR = (0, 0, 255)
CENTER_RADIUS = 6
ROW_COLOR = (0, 255, 0)
COL_COLOR = (0, 255, 255)
LINE_THICKNESS = 3
TEXT_COLOR = (255, 255, 255)
TEXT_SCALE = 0.55
TEXT_THICKNESS = 2


def draw_row_lines(img, row_lines, row_counts):
    h, w = img.shape[:2]
    for i, (a, b) in enumerate(row_lines):
        # a=0, so y = b (horizontal)
        y = int(round(b))
        y = max(0, min(h-1, y))
        cv2.line(img, (0, y), (w-1, y), ROW_COLOR, LINE_THICKNESS)
        if i < len(row_counts):
            cv2.putText(img, f"R{i}: {row_counts[i]}", (10, max(20, y - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, TEXT_SCALE, ROW_COLOR, TEXT_THICKNESS, cv2.LINE_AA)


def draw_column_lines(img, column_lines, column_counts):
    h, w = img.shape[:2]
    for i, (a, b) in enumerate(column_lines):
        # a=0, so x = b (vertical)
        x = int(round(b))
        x = max(0, min(w-1, x))
        cv2.line(img, (x, 0), (x, h-1), COL_COLOR, LINE_THICKNESS)
        if i < len(column_counts):
            cv2.putText(img, f"C{i}: {column_counts[i]}",
                        (max(10, x + 4), min(h - 10, 17 + 17 * (i % 3))),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, COL_COLOR, TEXT_THICKNESS, cv2.LINE_AA)


def annotate_side_debug(img, centers, geometry, boxes, save_path):
    vis = img.copy()
    if boxes is not None and len(boxes) > 0:
        for b in boxes:
            x1, y1, x2, y2 = map(int, b)
            cv2.rectangle(vis, (x1, y1), (x2, y2), BOX_COLOR, BOX_THICKNESS)

    for idx, (x, y) in enumerate(centers):
        cx, cy = int(round(x)), int(round(y))
        cv2.circle(vis, (cx, cy), CENTER_RADIUS, CENTER_COLOR, -1)
        cv2.drawMarker(vis, (cx, cy), (255, 255, 255), cv2.MARKER_CROSS, 10, 2)
        cv2.putText(vis, str(idx), (cx + 8, cy - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

    row_lines = geometry.get("row_lines", [])
    row_counts = geometry.get("row_counts", [])
    if row_lines:
        draw_row_lines(vis, row_lines, row_counts)

    column_lines = geometry.get("column_lines", [])
    col_counts = geometry.get("col_center_counts", {})
    if column_lines:
        col_list = [col_counts.get(i, 0) for i in range(len(column_lines))]
        draw_column_lines(vis, column_lines, col_list)

    lines = [
        f"rows: {geometry.get('num_rows', 0)}",
        f"cols: {geometry.get('num_cols', 0)}",
        f"row_counts: {row_counts}",
        f"col_counts: {geometry.get('col_counts', {})}",
    ]
    y = 28
    for line in lines:
        cv2.putText(vis, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
        y += 24

    cv2.imwrite(save_path, vis)
    logger.info(f"Annotated image saved: {save_path}")


# ================================================================
# Main
# ================================================================

def main():
    if not os.path.exists(SIDE_MODEL_PATH):
        logger.error(f"Model not found: {SIDE_MODEL_PATH}")
        sys.exit(1)

    logger.info("Loading YOLO model...")
    model = YOLO(SIDE_MODEL_PATH)

    for path, label in [(SIDE1_PATH, "Side-1"), (SIDE2_PATH, "Side-2")]:
        if not os.path.exists(path):
            logger.error(f"Image not found: {path}")
            continue

        img = cv2.imread(path)
        if img is None:
            logger.error(f"Could not read: {path}")
            continue

        logger.info(f"\nProcessing {label} ...")
        boxes, centers = infer_side(img, model)

        if boxes is None or len(boxes) < 2:
            logger.warning("Insufficient detections (< 2 boxes) – skipping geometry.")
            cv2.imwrite(f"{label}_raw.jpg", img)
            continue

        geom = compute_geometry(centers, boxes, is_left=True)

        logger.info(f"  num_rows   : {geom.get('num_rows', 0)}")
        logger.info(f"  num_cols   : {geom.get('num_cols', 0)}")
        logger.info(f"  row_counts : {geom.get('row_counts', [])}")
        logger.info(f"  col_counts : {geom.get('col_counts', {})}")
        logger.info(f"  col_center_counts : {geom.get('col_center_counts', {})}")

        out_name = os.path.splitext(os.path.basename(path))[0] + "_anchored_annotated.jpg"
        annotate_side_debug(img, centers, geom, boxes, out_name)

    logger.info("\nDone.")


if __name__ == "__main__":
    main()