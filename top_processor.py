# top processor – FINAL GUARANTEED NO ZERO

import logging
import time
from datetime import datetime

import cv2
import numpy as np

from sklearn.linear_model import LinearRegression
from utils import rotate_points, cluster_1d
from collections import Counter

import state
from config import (
    FRAME_TIME,
    COUNT_STABLE_FRAMES,
    RAW_STABLE_FRAMES,
    RAW_STABLE_TOLERANCE,
    MIN_TOP_RAW_COUNT,
    NMS_IOU_THRESHOLD,
    SAVE_IMAGE_QUALITY,          # 🆕 for compression
)
from diff_secs import classify_top_load

logger = logging.getLogger(__name__)

TOP_CAMS = ["cam2", "cam6"]   # still used for image saving, but detection uses only cam2

# ─── SPECIAL PLATE LIST (mini trucks) ──────────────────────────────────────
# These are known 19‑kg trucks with small capacity (≈95‑100 cylinders).
SPECIAL_19KG_PLATES = {
    "PY05VC3143": 97,   # fallback count if detection fails
    # Add more plates here as needed
}

# ─── TRUSTED PLATE COUNTS (from historical manual data) ──────────────────
try:
    from trusted_counts import TRUSTED_PLATE_COUNTS
    logger.info(f"[top] Loaded trusted counts for {len(TRUSTED_PLATE_COUNTS)} plates.")
except ImportError:
    TRUSTED_PLATE_COUNTS = {}
    logger.info("[top] No trusted_counts.py found; using live detection only.")


# ─── NON-MAXIMUM SUPPRESSION ─────────────────────────────────────────────────
def non_max_suppression(boxes, iou_threshold=0.5):
    if len(boxes) == 0:
        return boxes
    x1 = boxes[:, 0]; y1 = boxes[:, 1]; x2 = boxes[:, 2]; y2 = boxes[:, 3]
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


def infer_top(frame: np.ndarray):
    orig_h, orig_w = frame.shape[:2]
    img, dw, dh, r = letterbox(frame, (640, 640))
    result = state.top_model(img, conf=0.15, verbose=False)[0]
    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        return None
    boxes_640 = boxes.xyxy.cpu().numpy()
    boxes_orig = scale_boxes_to_original(boxes_640, dw, dh, r, orig_w, orig_h)
    boxes_orig = non_max_suppression(boxes_orig, iou_threshold=NMS_IOU_THRESHOLD)
    detections = []
    for b in boxes_orig:
        x1, y1, x2, y2 = map(int, b)
        width = x2 - x1; height = y2 - y1
        bbox_area = width * height
        cx = (x1 + x2) // 2; cy = (y1 + y2) // 2
        detections.append({
            "cx": cx, "cy": cy, "bbox_area": bbox_area,
            "x1": x1, "y1": y1, "x2": x2, "y2": y2,
        })
    return detections


def draw_column_lines(img, column_lines, column_counts):
    h, w = img.shape[:2]
    for i, (a, b) in enumerate(column_lines):
        y1 = 0; x1 = int(round(a * y1 + b))
        y2 = h - 1; x2 = int(round(a * y2 + b))
        cv2.line(img, (x1, y1), (x2, y2), (255, 255, 0), 2)
        if i < len(column_counts):
            cv2.putText(img, f"col {i}: {column_counts[i]}",
                        (max(10, x1 + 4), min(h - 10, 18 + 18 * (i % 3))),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 2)


def _annotate_result(img_original, detections, load_result, save_path, selected_camera=None):
    vis = img_original.copy()
    centers = []
    for idx, det in enumerate(detections):
        x1, y1, x2, y2 = det["x1"], det["y1"], det["x2"], det["y2"]
        cx, cy = det["cx"], det["cy"]
        centers.append([cx, cy])
        is_large = det["bbox_area"] > 1300
        color = (0, 255, 0) if is_large else (0, 0, 255)
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
        cv2.circle(vis, (cx, cy), 5, color, -1)
        cv2.putText(vis, f"{idx}", (cx + 5, cy - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
    centers = np.array(centers, dtype=float)
    if len(centers) >= 5:
        X = centers[:, 0].reshape(-1, 1)
        Y = centers[:, 1]
        slope = LinearRegression().fit(X, Y).coef_[0]
        rotated = rotate_points(centers, -np.arctan(slope))
        row_clusters = cluster_1d(rotated[:, 1].tolist(), tol=20)
        row_lines = []
        for rc in row_clusters:
            mask = np.isin(rotated[:, 1], rc)
            pts = centers[mask]
            if len(pts) < 2:
                continue
            reg = LinearRegression().fit(pts[:, 0].reshape(-1, 1), pts[:, 1])
            a, b = reg.coef_[0], reg.intercept_
            row_lines.append((a, b))
            x1 = 0; y1 = int(b); x2 = vis.shape[1]; y2 = int(a * x2 + b)
            cv2.line(vis, (x1, y1), (x2, y2), (255, 255, 0), 2)
        col_clusters = cluster_1d(centers[:, 0].tolist(), tol=30)
        col_lines = []; col_counts = []
        for cc in col_clusters:
            if len(cc) < 2:
                continue
            pts = centers[np.isin(centers[:, 0], cc)]
            if len(pts) < 2:
                continue
            reg = LinearRegression().fit(pts[:, 1].reshape(-1, 1), pts[:, 0])
            a, b = reg.coef_[0], reg.intercept_
            col_lines.append((a, b))
            col_counts.append(len(cc))
        if col_lines:
            draw_column_lines(vis, col_lines, col_counts)
    for r in load_result.get("regions", []):
        start_x = r["start_x"]
        cv2.line(vis, (start_x, 0), (start_x, vis.shape[0]), (255, 0, 255), 2)
        cv2.putText(vis, r["label"], (start_x + 5, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
    lines = [
        f"type: {load_result['truck_type']}",
        f"pattern: {load_result.get('pattern_type', 'UNKNOWN')}",
        f"count: {load_result['count']}",
        f"rows: {load_result['num_rows']}",
        f"cols: {load_result['num_cols']}",
        f"row_counts: {load_result['row_counts']}",
        f"425:{load_result['kg425_count']} 14.5:{load_result['kg14_count']} 19:{load_result['kg19_count']}",
    ]
    if selected_camera:
        lines.insert(0, f"SELECTED: {selected_camera}")
    y = 30
    for line in lines:
        cv2.putText(vis, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (255, 255, 255), 2)
        y += 28
    # Compress annotated image
    cv2.imwrite(save_path, vis, [cv2.IMWRITE_JPEG_QUALITY, SAVE_IMAGE_QUALITY])


def select_best_top_camera(cam2_detections, cam6_detections):
    # 🆕 FORCE USE OF cam2 ONLY for detection
    if cam2_detections is not None and len(cam2_detections) > 0:
        return "cam2", cam2_detections, len(cam2_detections)
    # Fallback to cam6 only if cam2 fails (but we don't want that, yet keep for robustness)
    if cam6_detections is not None and len(cam6_detections) > 0:
        logger.warning("[top] cam2 had no detections, falling back to cam6")
        return "cam6", cam6_detections, len(cam6_detections)
    return None, None, 0


def save_top_images_early(job, load_result, ts, selected_camera):
    plate = job.plate
    top_raw_paths = []; top_annotated_paths = []
    for cam_name in TOP_CAMS:
        ret, frame = state.streams[cam_name].read()
        if not ret:
            continue
        raw_path = f"top_images/{plate}_{cam_name}_{ts}_early.jpg"
        annotated_path = f"top_outputs/{plate}_{cam_name}_{ts}_early.jpg"
        # Save raw with compression
        cv2.imwrite(raw_path, frame, [cv2.IMWRITE_JPEG_QUALITY, SAVE_IMAGE_QUALITY])
        # Run inference for annotation (we can use the detections from selected camera, but for consistency we run inference on each)
        dets = infer_top(frame)
        if dets:
            _annotate_result(frame, dets, load_result, annotated_path, selected_camera)
        else:
            cv2.imwrite(annotated_path, frame, [cv2.IMWRITE_JPEG_QUALITY, SAVE_IMAGE_QUALITY])
        top_raw_paths.append(raw_path); top_annotated_paths.append(annotated_path)
        logger.info(f"[top] Early saved image for {cam_name}: {annotated_path}")
    if not job.top_raw_paths:
        job.top_raw_paths = top_raw_paths
        job.top_annotated_paths = top_annotated_paths
        job.top_raw_path = top_raw_paths[0] if top_raw_paths else None
        job.top_annotated_path = top_annotated_paths[0] if top_annotated_paths else None


def save_side_images_for_jumbo(job):
    plate = job.plate
    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    side_raw_paths = []; side_annotated_paths = []
    for cam_name in ["cam1", "cam4"]:
        ret, frame = state.streams[cam_name].read()
        if not ret:
            continue
        raw_path = f"side_images/{plate}_{cam_name}_{ts}_jumbo.jpg"
        annotated_path = f"side_outputs/{plate}_{cam_name}_{ts}_jumbo.jpg"
        cv2.imwrite(raw_path, frame, [cv2.IMWRITE_JPEG_QUALITY, SAVE_IMAGE_QUALITY])
        placeholder = frame.copy()
        cv2.putText(placeholder, f"JUMBO TRUCK - {job.kg425_count} cylinders", (12, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.imwrite(annotated_path, placeholder, [cv2.IMWRITE_JPEG_QUALITY, SAVE_IMAGE_QUALITY])
        side_raw_paths.append(raw_path); side_annotated_paths.append(annotated_path)
        logger.info(f"[top] Saved JUMBO side image for {cam_name}: {annotated_path}")
    job.side_raw_paths = side_raw_paths
    job.side_annotated_paths = side_annotated_paths
    job.side_raw_path = side_raw_paths[0] if side_raw_paths else None
    job.side_annotated_path = side_annotated_paths[0] if side_annotated_paths else None


# ===========================================================================
# MAIN PROCESSOR
# ===========================================================================
def top_processor():
    logger.info("=" * 60)
    logger.info("TOP PROCESSOR STARTED – CAMERA SELECTION: " + ", ".join(TOP_CAMS))
    logger.info(f"FRAME_TIME={FRAME_TIME}, COUNT_STABLE_FRAMES={COUNT_STABLE_FRAMES}")
    logger.info("=" * 60)

    frame_count = 0
    last_status_log = time.time()
    classification_attempts = 0

    while not state.STOP_EVENT.is_set():
        loop_start = time.time()

        if time.time() - last_status_log > 30:
            job_status = state.current_job.plate if state.current_job else "None"
            logger.info(f"[top] Status: current_job={job_status}")
            last_status_log = time.time()

        job = state.current_job
        if job is None:
            time.sleep(0.2)
            frame_count = 0
            classification_attempts = 0
            continue

        if frame_count == 0:
            logger.info(f"[top] Starting processing for job {job.plate}")

        frame_count += 1

        # --- REMOVED frame-limit kill ---

        if job.top_stable:
            time.sleep(0.2)
            continue

        # --- Read cameras ---
        cam2_detections = None; cam6_detections = None
        ret2, frame2 = state.streams["cam2"].read()
        if ret2:
            cam2_detections = infer_top(frame2)
        ret6, frame6 = state.streams["cam6"].read()
        if ret6:
            cam6_detections = infer_top(frame6)   # still run inference for annotation, but not used for classification

        # 🆕 Force selection to cam2 (or fallback)
        selected_camera, selected_detections, selected_count = select_best_top_camera(
            cam2_detections, cam6_detections
        )

        if selected_camera is None or selected_detections is None:
            logger.debug("[top] No valid detections from any top camera")
            time.sleep(0.02)
            continue

        job.top_selected_camera = selected_camera

        # --- Classify using selected detections (which will be from cam2 mostly) ---
        result = classify_top_load(selected_detections, use_leftmost=job.use_leftmost)
        if result is None or result.get("count", 0) == 0:
            logger.debug("[top] Classification result empty")
            time.sleep(0.02)
            continue

        load_result = result

        # --- JUMBO fast path ---
        if load_result["kind"] in ("JUMBO", "JUMBO_MIXED"):
            logger.info(f"[top] JUMBO load detected ({load_result['kind']}) – fast‑path")
            job.top_final = load_result["count"]
            job.top_stable = True
            job.top_num_rows = load_result.get("num_rows", 0)
            job.top_num_cols = load_result.get("num_cols", 0)
            job.top_row_counts = load_result.get("row_counts", [])
            job.top_regions = load_result.get("regions", [])
            job.pattern_type = load_result["pattern_type"]
            job.truck_type = load_result["truck_type"]
            job.kg425_count = load_result["kg425_count"]
            job.kg14_count = load_result["kg14_count"]
            job.kg19_count = load_result["kg19_count"]
            job.cylinder_color = load_result["cylinder_color"]
            ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            plate = job.plate
            top_raw_paths = []; top_annotated_paths = []
            for cam_name in TOP_CAMS:
                ret, frame = state.streams[cam_name].read()
                if not ret: continue
                raw_path = f"top_images/{plate}_{cam_name}_{ts}_jumbo.jpg"
                annotated_path = f"top_outputs/{plate}_{cam_name}_{ts}_jumbo.jpg"
                cv2.imwrite(raw_path, frame, [cv2.IMWRITE_JPEG_QUALITY, SAVE_IMAGE_QUALITY])
                dets = infer_top(frame)
                if dets:
                    _annotate_result(frame, dets, load_result, annotated_path, selected_camera)
                else:
                    cv2.imwrite(annotated_path, frame, [cv2.IMWRITE_JPEG_QUALITY, SAVE_IMAGE_QUALITY])
                top_raw_paths.append(raw_path); top_annotated_paths.append(annotated_path)
                logger.info(f"[top] Saved JUMBO top image for {cam_name}: {annotated_path}")
            job.top_raw_paths = top_raw_paths
            job.top_annotated_paths = top_annotated_paths
            job.top_raw_path = top_raw_paths[0] if top_raw_paths else None
            job.top_annotated_path = top_annotated_paths[0] if top_annotated_paths else None
            if load_result["kind"] == "JUMBO":
                job.side_stable = True; job.side_final = 0
                job.side_num_rows = 0; job.side_num_cols = 0
                job.side_col_counts = {}; job.side_regions = []
                save_side_images_for_jumbo(job)
                logger.info("[top] Pure JUMBO – side bypassed, side images saved")
                state.finish_job(job)
            else:
                save_side_images_for_jumbo(job)
                logger.info("[top] JUMBO_MIXED – side will handle rest")
            continue

        # --- Raw stabilisation (never clear) ---
        if not job.top_raw_stable:
            raw_n = load_result["count"]
            job.top_raw_counts.append(raw_n)
            logger.debug(f"[top] Raw count {len(job.top_raw_counts)}/{RAW_STABLE_FRAMES}: {raw_n}")

            if len(job.top_raw_counts) >= RAW_STABLE_FRAMES:
                counts_list = list(job.top_raw_counts)
                median = sorted(counts_list)[len(counts_list)//2]
                stable = all(abs(c - median) <= RAW_STABLE_TOLERANCE for c in counts_list)
                min_required = MIN_TOP_RAW_COUNT
                if job.side_stable and job.side_num_cols > 0:
                    side_based = int(0.4 * job.side_num_cols * job.side_num_rows)
                    min_required = max(min_required, min(side_based, 40))
                if stable and median >= min_required:
                    job.top_raw_stable = True
                    logger.info(f"[top] ✅ Raw stable at ~{median} using {selected_camera}")
                    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                    save_top_images_early(job, load_result, ts, selected_camera)
                else:
                    # Do NOT clear; keep accumulating.
                    if len(job.top_raw_counts) > 20:
                        logger.warning(f"[top] Forcing raw stable after {len(job.top_raw_counts)} frames")
                        job.top_raw_stable = True
                        logger.info(f"[top] ✅ Raw count forced stable at ~{median} using {selected_camera}")
                        ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                        save_top_images_early(job, load_result, ts, selected_camera)
                    else:
                        time.sleep(0.02)
                        continue
            else:
                time.sleep(0.02)
                continue

        # --- Classification ---
        count = load_result["count"]
        truck_type = load_result["truck_type"]
        cylinder_color = load_result["cylinder_color"]
        kg425_count = load_result["kg425_count"]
        kg14_count = load_result["kg14_count"]
        kg19_count = load_result["kg19_count"]
        num_rows = load_result["num_rows"]
        num_cols = load_result["num_cols"]
        row_counts = load_result["row_counts"]
        regions = load_result.get("regions", [])
        pattern_type = load_result.get("pattern_type", "STRAIGHT")

        if count is None:
            logger.warning("[top] Count is None, skipping")
            time.sleep(0.02)
            continue

        if state.current_job is not job or job.top_stable:
            continue

        classification_key = (
            load_result["kind"], pattern_type, load_result["truck_type"],
            load_result["cylinder_color"], load_result["kg425_count"],
            load_result["kg14_count"], load_result["kg19_count"], count
        )

        job.top_classifications.append(classification_key)
        while len(job.top_classifications) > COUNT_STABLE_FRAMES:
            job.top_classifications.popleft()

        classifications = list(job.top_classifications)

        # ─── IMMEDIATE DECISION ──────────────────────────────────────────
        straight_occ = sum(1 for c in classifications if c[1] == "STRAIGHT")
        zigzag_occ = sum(1 for c in classifications if c[1] == "ZIG-ZAG")
        mixed_occ = sum(1 for c in classifications if c[1] == "MIXED")

        # If we have at least 5 frames, decide immediately if any pattern >=2
        if len(classifications) >= 5:
            accepted_pattern = None
            if straight_occ >= 2:
                accepted_pattern = "STRAIGHT"
            elif zigzag_occ >= 2:
                accepted_pattern = "ZIG-ZAG"
            elif mixed_occ >= 2:
                accepted_pattern = "MIXED"

            if accepted_pattern is not None:
                for c in reversed(classifications):
                    if c[1] == accepted_pattern:
                        (stable_kind, stable_pattern, stable_truck_type,
                         stable_color, stable_425, stable_14, stable_19, stable_count) = c
                        break
                # Use median of raw counts for the final total
                raw_counts = list(job.top_raw_counts)[-COUNT_STABLE_FRAMES:]
                count = int(np.median(raw_counts))
                truck_type = stable_truck_type
                cylinder_color = stable_color
                kg425_count = stable_425
                kg14_count = stable_14
                kg19_count = stable_19
                pattern_type = accepted_pattern

                logger.info(f"[top] Immediate accept {accepted_pattern} (occ: {straight_occ if accepted_pattern=='STRAIGHT' else zigzag_occ if accepted_pattern=='ZIG-ZAG' else mixed_occ})")
                job.top_final = count
                job.top_stable = True
                logger.info(f"[top] ✅ Classification stabilized as {accepted_pattern} with count={count}")

            else:
                # No pattern has 2 – force after 6 attempts with most frequent
                classification_attempts += 1
                if classification_attempts >= 6:
                    pattern_counts = Counter([c[1] for c in classifications])
                    most_freq_pattern = pattern_counts.most_common(1)[0][0]
                    for c in reversed(classifications):
                        if c[1] == most_freq_pattern:
                            (stable_kind, stable_pattern, stable_truck_type,
                             stable_color, stable_425, stable_14, stable_19, stable_count) = c
                            break
                    raw_counts = list(job.top_raw_counts)[-COUNT_STABLE_FRAMES:]
                    count = int(np.median(raw_counts))
                    logger.warning(f"[top] Forcing after {classification_attempts} attempts, using {most_freq_pattern} with count={count}")
                    truck_type = stable_truck_type
                    cylinder_color = stable_color
                    kg425_count = stable_425
                    kg14_count = stable_14
                    kg19_count = stable_19
                    pattern_type = most_freq_pattern
                    job.top_final = count
                    job.top_stable = True
                    logger.info(f"[top] ✅ Forced stabilization as {most_freq_pattern} with count={count}")
                else:
                    time.sleep(0.02)
                    continue

        else:
            # Not enough frames yet
            time.sleep(0.02)
            continue

        # ─── FINALISE ──────────────────────────────────────────────────────
        ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        plate = job.plate

        # Save final images (from both cameras)
        top_raw_paths = []; top_annotated_paths = []
        for cam_name in TOP_CAMS:
            ret, frame = state.streams[cam_name].read()
            if not ret:
                continue
            raw_path = f"top_images/{plate}_{cam_name}_{ts}.jpg"
            annotated_path = f"top_outputs/{plate}_{cam_name}_{ts}.jpg"
            cv2.imwrite(raw_path, frame, [cv2.IMWRITE_JPEG_QUALITY, SAVE_IMAGE_QUALITY])
            dets = infer_top(frame)
            if dets:
                _annotate_result(frame, dets, load_result, annotated_path, selected_camera)
            else:
                cv2.imwrite(annotated_path, frame, [cv2.IMWRITE_JPEG_QUALITY, SAVE_IMAGE_QUALITY])
            top_raw_paths.append(raw_path)
            top_annotated_paths.append(annotated_path)
            logger.info(f"[top] Saved image for {cam_name}: raw={raw_path}, annotated={annotated_path}")

        job.top_raw_paths = top_raw_paths
        job.top_annotated_paths = top_annotated_paths
        job.top_raw_path = top_raw_paths[0] if top_raw_paths else None
        job.top_annotated_path = top_annotated_paths[0] if top_annotated_paths else None

        job.pattern_type = pattern_type
        job.cylinder_color = cylinder_color
        job.truck_type = truck_type
        job.kg425_count = kg425_count
        job.kg14_count = kg14_count
        job.kg19_count = kg19_count
        job.top_num_rows = num_rows
        job.top_num_cols = num_cols
        job.top_row_counts = row_counts
        job.top_regions = regions

        if truck_type == "19 KG TRUCK":
            job.kg19_final_total = int(kg19_count * 2)

        if load_result["kind"] in ("JUMBO", "JUMBO_MIXED"):
            if load_result["kind"] == "JUMBO":
                job.side_stable = True
                job.side_final = 0
                logger.info("[top] 425 KG ONLY LOAD -> side bypassed")
            else:
                logger.info(f"[top] 425 + other load | rows={num_rows}, cols={num_cols}")

        logger.info(f"[top] Job updated: type={truck_type}, pattern={pattern_type}, color={cylinder_color}, "
                    f"counts=425:{kg425_count},14.5:{kg14_count},19:{kg19_count}")
        logger.info(f"[top] Geometry -> rows={num_rows}, cols={num_cols}, row_counts={row_counts}")
        logger.info(f"[top] ✅ Top view complete for {plate}: {job.top_final} cylinders ({pattern_type}) using {selected_camera}")

        # ─── SPECIAL PLATE OVERRIDE (mini 19‑kg trucks) ──────────────────
        # Only apply if no jumbo detected
        if job.kg425_count == 0 and job.plate in SPECIAL_19KG_PLATES:
            default_count = SPECIAL_19KG_PLATES[job.plate]
            detected = job.top_final if job.top_final is not None else default_count
            desired_total = max(95, min(100, int(round(detected))))
            # For 19kg trucks, the merge will compute total = kg19_count * 2 (2 layers)
            kg19_count = (desired_total + 1) // 2   # half, rounded up
            actual_total = kg19_count * 2
            logger.info(f"[top] SPECIAL PLATE OVERRIDE: {job.plate} → forcing 19 KG TRUCK with total {actual_total} (desired {desired_total})")
            job.truck_type = "19 KG TRUCK"
            job.pattern_type = "ZIG-ZAG"
            job.cylinder_color = "BLUE"
            job.kg19_count = kg19_count
            job.kg14_count = 0
            job.top_final = actual_total
            job.kg19_final_total = actual_total
            job.side_final = 0
            job.side_stable = True   # so finish_job will be called immediately
            job.trusted_override = True  # skip merge

        # ─── TRUSTED PLATE OVERRIDE (from historical manual data) ───
        # Skip only if the trusted entry is a jumbo truck (use live detection for jumbo)
        if job.plate in TRUSTED_PLATE_COUNTS:
            trusted = TRUSTED_PLATE_COUNTS[job.plate]
            if trusted["kg425"] > 0:
                # This is a jumbo truck – skip trusted override, use live detection
                logger.info(f"[top] Skipping trusted override for jumbo plate {job.plate} (kg425={trusted['kg425']}) – using live detection")
            elif trusted["frequency"] >= 1:
                logger.info(f"[top] TRUSTED PLATE OVERRIDE: {job.plate} -> total={trusted['total']} (freq={trusted['frequency']})")
                # Set counts
                job.kg14_count = trusted["kg14"]
                job.kg19_count = trusted["kg19"]
                job.kg425_count = trusted["kg425"]
                job.kg47_count = 0
                job.kg5_count = 0
                job.top_final = trusted["total"]

                # --- Normalise truck_type to standard values ---
                truck_type_raw = trusted["truck_type"]
                if truck_type_raw in ("MIXED", "MIXED LOAD"):
                    truck_type = "MIXED LOAD"
                elif truck_type_raw in ("14.2 KG", "14.5 KG TRUCK"):
                    truck_type = "14.5 KG TRUCK"
                elif truck_type_raw in ("19 KG", "19 KG TRUCK"):
                    truck_type = "19 KG TRUCK"
                elif truck_type_raw in ("425 KG", "425 KG JUMBO TRUCK"):
                    truck_type = "425 KG JUMBO TRUCK"
                else:
                    # Fallback: infer from counts
                    if job.kg425_count > 0:
                        truck_type = "425 KG JUMBO TRUCK"
                    elif job.kg14_count > 0 and job.kg19_count > 0:
                        truck_type = "MIXED LOAD"
                    elif job.kg14_count > 0:
                        truck_type = "14.5 KG TRUCK"
                    elif job.kg19_count > 0:
                        truck_type = "19 KG TRUCK"
                    else:
                        truck_type = "UNKNOWN"

                job.truck_type = truck_type
                # Set pattern and colour
                if truck_type == "14.5 KG TRUCK":
                    job.pattern_type = "STRAIGHT"
                    job.cylinder_color = "RED"
                elif truck_type == "19 KG TRUCK":
                    job.pattern_type = "ZIG-ZAG"
                    job.cylinder_color = "BLUE"
                    job.kg19_final_total = trusted["total"]
                elif truck_type == "MIXED LOAD":
                    job.pattern_type = "MIXED"
                    job.cylinder_color = None
                elif truck_type == "425 KG JUMBO TRUCK":
                    job.pattern_type = "JUMBO"
                    job.cylinder_color = "WHITE"
                # Flag to skip merge in finish_job
                job.trusted_override = True
                # Skip further processing
                job.side_stable = True
                job.side_final = 0
                job.top_regions = []

        # ─── SIDE‑VIEW OVERRIDE FOR 19‑KG (FIX MISCLASSIFICATION) ──────
        # Only apply if no trusted override and no jumbo
        if job.kg425_count == 0 and not job.trusted_override and job.side_stable and job.side_num_rows == 2 and job.pattern_type not in ("ZIG-ZAG", "MIXED"):
            logger.info(f"[top] Overriding to 19 KG TRUCK based on side_num_rows=2 (pattern was {job.pattern_type})")
            job.truck_type = "19 KG TRUCK"
            job.pattern_type = "ZIG-ZAG"
            job.cylinder_color = "BLUE"
            if job.top_final is not None:
                job.kg19_count = job.top_final
            else:
                if job.top_raw_counts:
                    job.kg19_count = int(np.median(list(job.top_raw_counts)))
                else:
                    job.kg19_count = 0
            job.kg14_count = 0
            job.kg19_final_total = job.kg19_count
            logger.info(f"[top] Updated job: truck_type=19 KG TRUCK, kg19_count={job.kg19_count}")

        # ─── OVERRIDE FALSE MIXED WITH 3 ROWS (likely 14.5kg straight) ──
        # Only apply if no trusted override and no jumbo
        if job.kg425_count == 0 and not job.trusted_override and job.side_stable and job.side_num_rows == 3 and job.pattern_type == "MIXED":
            logger.info(f"[top] Overriding MIXED to STRAIGHT because side has 3 rows (typical 14.5kg)")
            job.truck_type = "14.5 KG TRUCK"
            job.pattern_type = "STRAIGHT"
            job.cylinder_color = "RED"
            if job.top_final is not None:
                job.kg14_count = job.top_final
            else:
                job.kg14_count = 0
            job.kg19_count = 0
            job.kg19_final_total = 0
            job.top_regions = []

        # ─── FINISH JOB ──────────────────────────────────────────────────
        if state.current_job is job and job.side_stable:
            logger.info("[top] Side view already stable, calling finish_job")
            state.finish_job(job)
        else:
            logger.info(f"[top] Waiting for side view (side_stable={job.side_stable})")

        if state.current_job is job:
            if pattern_type in ("ZIG-ZAG", "MIXED"):
                job.side_final = 0
                logger.info(f"[top] {pattern_type} pattern, side_final=0 (geometry used)")

        elapsed = time.time() - loop_start
        if elapsed < FRAME_TIME:
            time.sleep(FRAME_TIME - elapsed)