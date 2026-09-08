import logging
import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from collections import Counter

import cv2
import numpy as np
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.linear_model import LinearRegression

import state
from side_utils import summarize_side_geometry_simple
from config import (
    FRAME_TIME,
    COUNT_STABLE_FRAMES,
    RAW_STABLE_FRAMES,
    RAW_STABLE_TOLERANCE,
    MIN_SIDE_RAW_COLUMNS,
    NMS_IOU_THRESHOLD,
    SAVE_IMAGE_QUALITY,          # 🆕 for compression
)

logger = logging.getLogger(__name__)

# Debug counters
frame_counter = 0
last_log_time = time.time()

# Local geometry defaults
INFER_SIZE = 640
MAX_ROWS = 3

# Side cameras (primary + fallback)
SIDE_CAM_PRIMARY = "cam1"
SIDE_CAM_FALLBACK = "cam4"

# Selection thresholds
MIN_SIDE_DETECTIONS = 30
MIN_SIDE_COLUMNS = 10

# 🆕 Position-based selection thresholds
# Cam 1 should have 20-30% empty space on the left when truck is properly placed
EMPTY_SPACE_MIN = 15  # Minimum empty space % for Cam 1 to be considered good
EMPTY_SPACE_MAX = 40  # Maximum empty space % for Cam 1 to be considered good

# Column geometry tuning
MAX_CENTERS_PER_COLUMN = 3
MAX_COLUMN_ANGLE_DEG = 10.0
MAX_COLUMN_SLOPE = np.tan(np.radians(MAX_COLUMN_ANGLE_DEG))
CENTER_LINE_TOLERANCE = 0.25
BOX_LINE_MARGIN = 5

BOX_COLOR = (0, 255, 0)
CENTER_COLOR = (0, 165, 255)
ROW_LINE_COLOR = (0, 255, 0)
COL_LINE_COLOR = (0, 255, 255)
REGION_LINE_COLOR = (255, 0, 255)
TEXT_COLOR = (255, 255, 255)

BOX_THICKNESS = 2
CENTER_RADIUS = 4
TEXT_SCALE = 0.55
TEXT_THICKNESS = 2
LINE_THICKNESS = 2


# ─── LETTERBOX (keeps aspect ratio) ────────────────────────────────────────
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


# =============================================================================
# LOCAL HELPERS
# =============================================================================
def rotate_points(points: np.ndarray, angle_rad: float) -> np.ndarray:
    if points is None or len(points) == 0:
        return np.empty((0, 2), dtype=float)
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    rot = np.array([[c, -s], [s, c]], dtype=float)
    return points @ rot.T


def is_raw_count_stable(counts, stable_frames: int, tolerance: int) -> bool:
    if counts is None or len(counts) < stable_frames:
        return False
    window = list(counts)[-stable_frames:]
    return (max(window) - min(window)) <= tolerance


def is_raw_stable_median(counts, stable_frames: int, tolerance: int) -> bool:
    if len(counts) < stable_frames:
        return False
    counts_list = list(counts)[-stable_frames:]
    median = sorted(counts_list)[len(counts_list) // 2]
    return all(abs(c - median) <= tolerance for c in counts_list)


def boxes_to_centers(boxes: np.ndarray) -> np.ndarray:
    if boxes is None or len(boxes) == 0:
        return np.empty((0, 2), dtype=float)
    return np.stack([(boxes[:, 0] + boxes[:, 2]) / 2.0,
                     (boxes[:, 1] + boxes[:, 3]) / 2.0], axis=1)


def non_max_suppression(boxes, iou_threshold=0.5):
    """Remove overlapping boxes (duplicate detections)."""
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
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)
        order = order[1:][iou < iou_threshold]
    return boxes[keep]


def _safe_int(v) -> int:
    return int(round(float(v)))


def _scale_region_x(region: dict, scale_x: float) -> dict:
    return {
        "label": region.get("label"),
        "start_x": int(round(region.get("start_x", 0) * scale_x)),
        "end_x": int(round(region.get("end_x", 0) * scale_x)),
    }


def scale_regions_to_original(regions: list, orig_w: int, infer_size: int = INFER_SIZE) -> list:
    if not regions:
        return []
    max_end_x = max(int(r.get("end_x", 0)) for r in regions)
    if max_end_x > infer_size or orig_w == infer_size:
        return [dict(r) for r in regions]
    scale_x = orig_w / float(infer_size)
    return [_scale_region_x(r, scale_x) for r in regions]


# ─── 🆕 CALCULATE EMPTY SPACE PERCENTAGE ──────────────────────────────────
def calculate_empty_space_percentage(centers, frame_width):
    """
    Calculate the percentage of empty space on the left side of the frame.
    Returns: empty_space_pct (0-100)
    """
    if centers is None or len(centers) == 0:
        return 100.0  # No detections means 100% empty

    # Get the leftmost cylinder center x-coordinate
    leftmost_x = min(centers[:, 0]) if isinstance(centers, np.ndarray) else min(c[0] for c in centers)

    # Calculate empty space percentage
    empty_space_pct = (leftmost_x / frame_width) * 100

    return empty_space_pct


# =============================================================================
# INFERENCE (with NMS)
# =============================================================================
def infer_side(frame: np.ndarray):
    orig_h, orig_w = frame.shape[:2]
    infer_img, dw, dh, r = letterbox(frame, (INFER_SIZE, INFER_SIZE))
    results = state.side_model(infer_img, conf=0.15, verbose=False)
    boxes_640 = results[0].boxes.xyxy.cpu().numpy() if results and results[0].boxes is not None else np.empty((0, 4), dtype=float)
    if len(boxes_640) == 0:
        return None, None, [], orig_w, orig_h
    boxes = scale_boxes_to_original(boxes_640, dw, dh, r, orig_w, orig_h)
    boxes = non_max_suppression(boxes, iou_threshold=NMS_IOU_THRESHOLD)
    centers = boxes_to_centers(boxes)
    pixel_heights = [float(b[3] - b[1]) for b in boxes]
    return boxes, centers, pixel_heights, orig_w, orig_h


# =============================================================================
# ANNOTATION HELPERS
# =============================================================================
def draw_row_lines(img: np.ndarray, row_lines: List[Tuple[float, float]], row_counts: List[int]) -> None:
    h, w = img.shape[:2]
    for i, (a, b) in enumerate(row_lines):
        x1 = 0
        y1 = int(round(a * x1 + b))
        x2 = w - 1
        y2 = int(round(a * x2 + b))
        cv2.line(img, (x1, y1), (x2, y2), ROW_LINE_COLOR, LINE_THICKNESS)
        if i < len(row_counts):
            cv2.putText(img, f"row {i}: {row_counts[i]}", (10, max(20, y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, TEXT_SCALE, ROW_LINE_COLOR, TEXT_THICKNESS, cv2.LINE_AA)


def draw_column_lines(img: np.ndarray, column_lines: List[Tuple[float, float]], column_counts: List[int]) -> None:
    h, w = img.shape[:2]
    for i, (a, b) in enumerate(column_lines):
        y1 = 0
        x1 = int(round(a * y1 + b))
        y2 = h - 1
        x2 = int(round(a * y2 + b))
        cv2.line(img, (x1, y1), (x2, y2), COL_LINE_COLOR, LINE_THICKNESS)
        if i < len(column_counts):
            cv2.putText(img, f"col {i}: {column_counts[i]}",
                        (max(10, x1 + 4), min(h - 10, 17 + 17 * (i % 3))),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, COL_LINE_COLOR, TEXT_THICKNESS, cv2.LINE_AA)


def annotate_side_geometry(
    img_original: np.ndarray,
    centers: np.ndarray,
    geometry: dict,
    regions: list,
    save_path: str,
    draw_boxes: bool = True,
    boxes: Optional[np.ndarray] = None,
    selected_camera: str = None,
    empty_space_pct: float = None,
) -> None:
    """Draw everything on the original image."""
    vis = img_original.copy()
    centers = np.asarray(centers) if centers is not None else np.empty((0, 2), dtype=float)

    if draw_boxes and boxes is not None and len(boxes) > 0:
        for b in boxes:
            x1, y1, x2, y2 = map(_safe_int, b)
            cv2.rectangle(vis, (x1, y1), (x2, y2), BOX_COLOR, BOX_THICKNESS)

    for idx, (x, y) in enumerate(centers):
        x_i, y_i = _safe_int(x), _safe_int(y)
        cv2.circle(vis, (x_i, y_i), CENTER_RADIUS, CENTER_COLOR, -1)
        cv2.putText(vis, str(idx), (x_i + 4, y_i - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, TEXT_COLOR, 1, cv2.LINE_AA)

    row_lines = geometry.get("row_lines", [])
    column_lines = geometry.get("column_lines", [])
    row_counts = geometry.get("row_counts", [])
    col_center_counts_dict = geometry.get("col_center_counts", {})

    if row_lines:
        draw_row_lines(vis, row_lines, row_counts)

    if column_lines:
        column_counts = [col_center_counts_dict.get(i, 0) for i in range(len(column_lines))]
        draw_column_lines(vis, column_lines, column_counts)

    # Region boundaries/labels
    if regions:
        for r in regions:
            start_x = int(r["start_x"])
            cv2.line(vis, (start_x, 0), (start_x, vis.shape[0] - 1), REGION_LINE_COLOR, LINE_THICKNESS)
            cv2.putText(vis, r["label"], (start_x + 5, 75),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, REGION_LINE_COLOR, 2, cv2.LINE_AA)

    # Summary text
    lines = []
    if selected_camera:
        lines.append(f"SELECTED: {selected_camera}")
    if empty_space_pct is not None:
        lines.append(f"Empty Space: {empty_space_pct:.1f}%")
    if regions:
        lines.append(f"MIXED REGIONS: {len(regions)}")
        for r in regions:
            lines.append(f"{r['label']} rows={r['num_rows']} cols={r['num_cols']} {r['col_counts']}")
    else:
        lines.extend([
            f"rows: {geometry.get('num_rows', 0)}",
            f"cols: {geometry.get('num_cols', 0)}",
            f"row_counts: {geometry.get('row_counts', [])}",
            f"col_center_counts (raw): {col_center_counts_dict}",
            f"col_counts (final): {geometry.get('col_counts', {})}",
        ])

    y = 28
    for line in lines:
        cv2.putText(vis, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, TEXT_COLOR, 2, cv2.LINE_AA)
        y += 24

    # Compress annotated image
    cv2.imwrite(save_path, vis, [cv2.IMWRITE_JPEG_QUALITY, SAVE_IMAGE_QUALITY])


# ─── SAVE IMAGES EARLY ─────────────────────────────────────────────────────
def save_side_images_early(job, geometry, ts, selected_camera, empty_space_pct=None):
    plate = job.plate
    side_raw_paths = []
    side_annotated_paths = []

    for cam_name in [SIDE_CAM_PRIMARY, SIDE_CAM_FALLBACK]:
        ret, frame = state.streams[cam_name].read()
        if not ret:
            continue

        raw_path = f"side_images/{plate}_{cam_name}_{ts}_early.jpg"
        annotated_path = f"side_outputs/{plate}_{cam_name}_{ts}_early.jpg"
        cv2.imwrite(raw_path, frame, [cv2.IMWRITE_JPEG_QUALITY, SAVE_IMAGE_QUALITY])

        boxes, centers, _, _, _ = infer_side(frame)
        if boxes is not None and len(boxes) > 0:
            geom = summarize_side_geometry_simple(centers, boxes)
            annotate_side_geometry(frame, centers, geom, [], annotated_path, draw_boxes=True, boxes=boxes, selected_camera=selected_camera, empty_space_pct=empty_space_pct)
        else:
            cv2.imwrite(annotated_path, frame, [cv2.IMWRITE_JPEG_QUALITY, SAVE_IMAGE_QUALITY])

        side_raw_paths.append(raw_path)
        side_annotated_paths.append(annotated_path)
        logger.info(f"[side] Early saved image for {cam_name}: {annotated_path}")

    if not job.side_raw_paths:
        job.side_raw_paths = side_raw_paths
        job.side_annotated_paths = side_annotated_paths
        job.side_raw_path = side_raw_paths[0] if side_raw_paths else None
        job.side_annotated_path = side_annotated_paths[0] if side_annotated_paths else None


# ─── SAVE FINAL IMAGES ─────────────────────────────────────────────────────
def save_side_images_final(job, geometry, ts, frames_for_annot, selected_camera, empty_space_pct=None):
    plate = job.plate
    side_raw_paths = []
    side_annotated_paths = []

    for cam_name in [SIDE_CAM_PRIMARY, SIDE_CAM_FALLBACK]:
        ret, frame = state.streams[cam_name].read()
        if not ret:
            continue

        raw_path = f"side_images/{plate}_{cam_name}_{ts}.jpg"
        annotated_path = f"side_outputs/{plate}_{cam_name}_{ts}.jpg"
        cv2.imwrite(raw_path, frame, [cv2.IMWRITE_JPEG_QUALITY, SAVE_IMAGE_QUALITY])

        boxes, centers, _, _, _ = infer_side(frame)
        if boxes is not None and len(boxes) > 0:
            geom = summarize_side_geometry_simple(centers, boxes)
            annotate_side_geometry(frame, centers, geom, [], annotated_path, draw_boxes=True, boxes=boxes, selected_camera=selected_camera, empty_space_pct=empty_space_pct)
        else:
            cv2.imwrite(annotated_path, frame, [cv2.IMWRITE_JPEG_QUALITY, SAVE_IMAGE_QUALITY])

        side_raw_paths.append(raw_path)
        side_annotated_paths.append(annotated_path)
        logger.info(f"[side] Saved image for {cam_name}: raw={raw_path}, annotated={annotated_path}")

    job.side_raw_paths = side_raw_paths
    job.side_annotated_paths = side_annotated_paths
    job.side_raw_path = side_raw_paths[0] if side_raw_paths else None
    job.side_annotated_path = side_annotated_paths[0] if side_annotated_paths else None

    # Handle regions if needed
    if frames_for_annot:
        frame_to_save, centers_for_annot, boxes_for_annot, cam_src = frames_for_annot[0]
        orig_w = frame_to_save.shape[1]
        scaled_top_regions = scale_regions_to_original(
            getattr(job, "top_regions", []) or [],
            orig_w
        )
        if scaled_top_regions:
            side_regions = []
            for region in scaled_top_regions:
                side_regions.append({
                    "label": region["label"],
                    "start_x": region["start_x"],
                    "end_x": region["end_x"],
                    "num_rows": geometry["num_rows"],
                    "num_cols": geometry["num_cols"],
                    "num_centers_in_lastrow": geometry.get("num_centers_in_lastrow", 0),
                    "col_counts": geometry["col_counts"],
                    "row_counts": geometry["row_counts"],
                })
            job.side_regions = side_regions
            job.side_num_rows = geometry["num_rows"]
            job.side_num_cols = geometry["num_cols"]
            job.side_num_centers_in_lastrow = geometry.get("num_centers_in_lastrow", 0)
            job.side_col_counts = geometry["col_counts"]
        else:
            job.side_num_rows = geometry["num_rows"]
            job.side_num_cols = geometry["num_cols"]
            job.side_num_centers_in_lastrow = geometry.get("num_centers_in_lastrow", 0)
            job.side_col_counts = geometry["col_counts"]
            job.side_regions = []


# =============================================================================
# MAIN PROCESSOR
# =============================================================================
def side_processor():
    logger.info("=" * 60)
    logger.info("SIDE PROCESSOR STARTED – PRIMARY: " + SIDE_CAM_PRIMARY + ", FALLBACK: " + SIDE_CAM_FALLBACK)
    logger.info(f"SELECTION CRITERIA: detections >= {MIN_SIDE_DETECTIONS}, columns >= {MIN_SIDE_COLUMNS}")
    logger.info(f"POSITION-BASED SELECTION: empty space {EMPTY_SPACE_MIN}%-{EMPTY_SPACE_MAX}% for primary camera")
    logger.info(f"FRAME_TIME={FRAME_TIME}, COUNT_STABLE_FRAMES={COUNT_STABLE_FRAMES}")
    logger.info("=" * 60)

    loop_count = 0
    no_job_logged = False
    max_side_loops = 300

    while not state.STOP_EVENT.is_set():
        loop_start = time.time()
        loop_count += 1

        if loop_count % 100 == 0:
            logger.debug(f"[side] Alive (loop {loop_count})")

        job = state.current_job
        if job is None:
            if not no_job_logged:
                logger.debug("[side] No active job, waiting...")
                no_job_logged = True
            time.sleep(0.2)
            continue

        no_job_logged = False

        if loop_count % 50 == 0:
            logger.info(f"[side] Active job: plate={job.plate}, side_stable={job.side_stable}, top_stable={job.top_stable}")

        if job.side_stable:
            time.sleep(0.2)
            continue

        truck_type = job.truck_type

        # Bypass for 425-only
        if truck_type == "425 KG JUMBO TRUCK":
            logger.info(f"[side] Skipping side view for {truck_type}, marking as complete")
            ret, frame = state.streams["cam4"].read()
            if not ret:
                logger.warning("[side] Failed to read frame for bypass path")
                time.sleep(0.02)
                continue

            ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            plate = job.plate
            raw_path = f"side_images/{plate}_{ts}.jpg"
            annotated_path = f"side_outputs/{plate}_{ts}.jpg"
            cv2.imwrite(raw_path, frame, [cv2.IMWRITE_JPEG_QUALITY, SAVE_IMAGE_QUALITY])
            placeholder = frame.copy()
            cv2.putText(placeholder, f"SKIPPED: {truck_type}", (12, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, TEXT_COLOR, 2, cv2.LINE_AA)
            cv2.imwrite(annotated_path, placeholder, [cv2.IMWRITE_JPEG_QUALITY, SAVE_IMAGE_QUALITY])

            if state.current_job is job:
                job.side_stable = True
                job.side_final = 0
                job.side_num_rows = 0
                job.side_num_cols = 0
                job.side_num_centers_in_lastrow = 0
                job.side_col_counts = {}
                job.side_regions = []
                job.side_raw_path = raw_path
                job.side_annotated_path = annotated_path
                if job.top_stable:
                    state.finish_job(job)
            continue

        # ─── STEP 1: Try Primary Camera (Cam 1) ──────────────────────────
        selected_camera = None
        selected_boxes = None
        selected_centers = None
        selected_geometry = None
        selected_frame = None
        empty_space_pct = None

        ret1, frame1 = state.streams[SIDE_CAM_PRIMARY].read()
        if ret1:
            boxes1, centers1, pixel_heights1, orig_w1, orig_h1 = infer_side(frame1)
            if boxes1 is not None and len(boxes1) >= 3:
                # Remove 425 cylinders if present
                working_centers1 = centers1
                working_boxes1 = boxes1
                if job.kg425_count and job.kg425_count > 0 and len(centers1) > job.kg425_count:
                    heights1 = np.array(pixel_heights1, dtype=float)
                    jumbo_idx1 = np.argsort(heights1)[-job.kg425_count:]
                    keep_mask1 = np.ones(len(centers1), dtype=bool)
                    keep_mask1[jumbo_idx1] = False
                    working_centers1 = centers1[keep_mask1]
                    working_boxes1 = boxes1[keep_mask1]
                    logger.info(f"[side] {SIDE_CAM_PRIMARY}: removed {job.kg425_count} jumbo centers")

                # ─── 🆕 Calculate empty space percentage ──────────────────
                empty_space_pct = calculate_empty_space_percentage(working_centers1, orig_w1)
                logger.debug(f"[side] {SIDE_CAM_PRIMARY} empty space: {empty_space_pct:.1f}%")

                geom1 = summarize_side_geometry_simple(working_centers1, working_boxes1, is_left=job.use_leftmost)
                if geom1 and geom1.get("num_rows", 0) > 0:
                    # ─── 🆕 Position-based decision ──────────────────────
                    # If empty space is within the ideal range (15-40%),
                    # Cam 1 is properly placed – use it.
                    if EMPTY_SPACE_MIN <= empty_space_pct <= EMPTY_SPACE_MAX:
                        selected_camera = SIDE_CAM_PRIMARY
                        selected_boxes = working_boxes1
                        selected_centers = working_centers1
                        selected_geometry = geom1
                        selected_frame = frame1
                        logger.info(f"[side] ✅ Using PRIMARY camera {SIDE_CAM_PRIMARY} (empty space: {empty_space_pct:.1f}%)")
                    else:
                        # Also check if Cam 1 has enough detections even if empty space is off
                        det_count1 = len(working_centers1)
                        col_count1 = geom1.get("num_cols", 0)
                        if det_count1 >= MIN_SIDE_DETECTIONS and col_count1 >= MIN_SIDE_COLUMNS:
                            # Still use Cam 1 if it has enough detections
                            selected_camera = SIDE_CAM_PRIMARY
                            selected_boxes = working_boxes1
                            selected_centers = working_centers1
                            selected_geometry = geom1
                            selected_frame = frame1
                            logger.info(f"[side] ✅ Using PRIMARY camera {SIDE_CAM_PRIMARY} (detections={det_count1}, cols={col_count1}) despite empty space: {empty_space_pct:.1f}%")
                        else:
                            logger.debug(f"[side] {SIDE_CAM_PRIMARY} below thresholds: empty={empty_space_pct:.1f}%, detections={det_count1}, cols={col_count1}")

        # ─── STEP 2: Fallback to Cam 4 if needed ──────────────────────────
        if selected_camera is None:
            ret4, frame4 = state.streams[SIDE_CAM_FALLBACK].read()
            if ret4:
                boxes4, centers4, pixel_heights4, orig_w4, orig_h4 = infer_side(frame4)
                if boxes4 is not None and len(boxes4) >= 3:
                    working_centers4 = centers4
                    working_boxes4 = boxes4
                    if job.kg425_count and job.kg425_count > 0 and len(centers4) > job.kg425_count:
                        heights4 = np.array(pixel_heights4, dtype=float)
                        jumbo_idx4 = np.argsort(heights4)[-job.kg425_count:]
                        keep_mask4 = np.ones(len(centers4), dtype=bool)
                        keep_mask4[jumbo_idx4] = False
                        working_centers4 = centers4[keep_mask4]
                        working_boxes4 = boxes4[keep_mask4]
                        logger.info(f"[side] {SIDE_CAM_FALLBACK}: removed {job.kg425_count} jumbo centers")

                    geom4 = summarize_side_geometry_simple(working_centers4, working_boxes4, is_left=job.use_leftmost)
                    if geom4 and geom4.get("num_rows", 0) > 0:
                        selected_camera = SIDE_CAM_FALLBACK
                        selected_boxes = working_boxes4
                        selected_centers = working_centers4
                        selected_geometry = geom4
                        selected_frame = frame4
                        det_count4 = len(working_centers4)
                        col_count4 = geom4.get("num_cols", 0)
                        logger.info(f"[side] ⚠️ Using FALLBACK camera {SIDE_CAM_FALLBACK} (detections={det_count4}, cols={col_count4})")

        # ─── STEP 3: Emergency fallback – try both and use the better one ──
        if selected_camera is None:
            logger.warning("[side] Both cameras failed. Trying emergency fallback...")
            # Try Cam 1 again, but without strict thresholds
            if ret1 and boxes1 is not None and len(boxes1) >= 3:
                geom1 = summarize_side_geometry_simple(working_centers1, working_boxes1, is_left=job.use_leftmost)
                if geom1 and geom1.get("num_rows", 0) > 0:
                    selected_camera = SIDE_CAM_PRIMARY
                    selected_boxes = working_boxes1
                    selected_centers = working_centers1
                    selected_geometry = geom1
                    selected_frame = frame1
                    logger.info(f"[side] 🔄 Emergency: using {SIDE_CAM_PRIMARY} (only available)")
            elif ret4 and boxes4 is not None and len(boxes4) >= 3:
                geom4 = summarize_side_geometry_simple(working_centers4, working_boxes4, is_left=job.use_leftmost)
                if geom4 and geom4.get("num_rows", 0) > 0:
                    selected_camera = SIDE_CAM_FALLBACK
                    selected_boxes = working_boxes4
                    selected_centers = working_centers4
                    selected_geometry = geom4
                    selected_frame = frame4
                    logger.info(f"[side] 🔄 Emergency: using {SIDE_CAM_FALLBACK} (only available)")

        if selected_camera is None or selected_geometry is None:
            logger.debug("[side] No valid geometry from any side camera")
            time.sleep(0.02)
            continue

        # Store which camera was selected
        job.side_selected_camera = selected_camera

        geometry = selected_geometry

        # ─── RAW STABILISATION ──────────────────────────────────────────────
        if not job.side_raw_stable:
            raw_n = len(geometry.get("col_counts", {}))
            job.side_raw_counts.append(raw_n)
            logger.debug(f"[side] Raw count {len(job.side_raw_counts)}/{RAW_STABLE_FRAMES}: {raw_n} (from {selected_camera})")

            if len(job.side_raw_counts) >= RAW_STABLE_FRAMES:
                counts_list = list(job.side_raw_counts)
                median = sorted(counts_list)[len(counts_list) // 2]
                stable = all(abs(c - median) <= RAW_STABLE_TOLERANCE for c in counts_list)

                min_required = MIN_SIDE_RAW_COLUMNS
                if job.top_stable and job.top_num_cols > 0:
                    min_required = max(min_required, int(0.6 * job.top_num_cols))

                if stable and median >= min_required:
                    job.side_raw_stable = True
                    logger.info(f"[side] ✅ Raw count stabilized at ~{median} (counts: {counts_list}) using {selected_camera}")
                    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                    save_side_images_early(job, geometry, ts, selected_camera, empty_space_pct)

                    row_count = geometry.get("num_rows", 0)
                    if row_count == 0:
                        if truck_type == "19 KG TRUCK" or truck_type == "MIXED LOAD":
                            row_count = 2
                        else:
                            row_count = 3
                        logger.warning(f"[side] num_rows was 0, falling back to {row_count} based on truck type")

                    logger.info(f"[side] ✅ Side view complete with row count = {row_count}")

                    ts_final = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                    save_side_images_final(job, geometry, ts_final, [(selected_frame, selected_centers, selected_boxes, selected_camera)], selected_camera, empty_space_pct)

                    job.side_final = int(row_count)
                    job.side_stable = True
                    job.side_num_rows = geometry.get("num_rows", row_count)
                    job.side_num_cols = geometry.get("num_cols", 0)
                    job.side_num_centers_in_lastrow = geometry.get("num_centers_in_lastrow", 0)
                    job.side_col_counts = geometry.get("col_counts", {})

                    logger.info(f"[side] ✅ Side view complete for {job.plate}: "
                                f"side_num_rows={job.side_num_rows}, side_num_cols={job.side_num_cols} (using {selected_camera})")

                    if state.current_job is job and job.top_stable:
                        logger.info("[side] Top already stable, finishing job")
                        state.finish_job(job)
                    else:
                        logger.info(f"[side] Waiting for top view (top_stable={job.top_stable})")
                else:
                    logger.debug(f"[side] Reset: median={median}, min_required={min_required}, stable={stable}")
                    job.side_raw_counts.clear()
                    continue

            time.sleep(0.02)
            continue

        # ─── FALLBACK: if raw stabilisation never happens ──────────────────
        if loop_count > max_side_loops and not job.side_stable:
            logger.warning(f"[side] ⏰ Side processor timeout after {loop_count} loops, forcing completion")
            job.side_stable = True
            job.side_final = 0
            job.side_num_rows = 0
            job.side_num_cols = 0
            job.side_col_counts = {}
            job.side_regions = []
            if state.current_job is job and job.top_stable:
                state.finish_job(job)

        elapsed = time.time() - loop_start
        if elapsed < FRAME_TIME:
            time.sleep(FRAME_TIME - elapsed)