import logging
import time
from datetime import datetime
from typing import Optional
from collections import defaultdict

import cv2
import numpy as np
import requests

from config import (
    CAM_RECORD_TYPE,
    CAM_USE_LEFTMOST,
    PLATE_CONF,
    OCR_URL,
    OCR_TIMEOUT,
    ANPR_FRAME_TIME,
    FRONT_STABLE_FRAMES,
    PLATE_OCR_STABLE_FRAMES,
    PLATE_OCR_TIMEOUT,
    GLOBAL_COOLDOWN_SEC,
    COOLDOWN_SEC,
    SAVE_IMAGE_QUALITY,
)
import state
from plate_validation import validate_plate, correct_plate, strip_india_marking
from utils import plate_similarity

# ─── Import recent plate cache from state ──────────────────────────────
from state import (
    recent_plates,
    recent_plates_lock,
    RECENT_PLATE_WINDOW_SEC,
    RECENT_PLATE_SIMILARITY_THRESHOLD,
)

# ─── Import special and trusted plates ────────────────────────────────
SPECIAL_19KG_PLATES = {
    "PY05VC3143": 97,
    # Add more as needed
}
try:
    from trusted_counts import TRUSTED_PLATE_COUNTS
except ImportError:
    TRUSTED_PLATE_COUNTS = {}

logger = logging.getLogger(__name__)


def plate_monitor(cam_name: str):
    """
    Two‑stage pipeline:
    1. FRONT/REAR classification – first camera to see FRONT for N frames wins.
    2. Plate detection + OCR – winner camera reads plate, stabilises for M identical results,
       then creates a job. Other cameras are suppressed during job + global cooldown.
    """
    # ─── Camera‑specific overrides ──────────────────────────────────────
    if cam_name == "cam5":
        # ENTRY: faster, more permissive to capture more trucks
        front_stable_frames = 7
        plate_ocr_stable_frames = 6
        plate_conf = 0.40
        cooldown_sec = 30
        fallback_timeout = 25
    else:
        # EXIT: stable, conservative settings
        front_stable_frames = FRONT_STABLE_FRAMES  # 15
        plate_ocr_stable_frames = PLATE_OCR_STABLE_FRAMES  # 15
        plate_conf = PLATE_CONF                # 0.60
        cooldown_sec = COOLDOWN_SEC            # 30 (you changed this)
        fallback_timeout = 15

    logger.info("=" * 60)
    logger.info(f"[{cam_name}] MONITOR STARTED - Type: {CAM_RECORD_TYPE[cam_name]}")
    logger.info(f"[{cam_name}] FRONT_STABLE_FRAMES={front_stable_frames}, PLATE_OCR_STABLE_FRAMES={plate_ocr_stable_frames}, PLATE_CONF={plate_conf}")
    logger.info("=" * 60)

    front_consecutive = 0
    last_orientation_result = None
    orientation_log_counter = 0

    plate_counter = defaultdict(int)
    stabilized_plate = None
    plate_reading_started_at = 0.0
    plate_reading_deadline = 0.0
    last_plate_detection_time = 0.0
    plate_change_count = 0

    last_frame_time = time.time()
    frame_counter = 0
    last_status_log = time.time()

    fallback_created = False

    while not state.STOP_EVENT.is_set():
        now = time.time()
        frame_counter += 1

        # Periodic status log
        if now - last_status_log > 30:
            with state.active_camera_lock:
                cooldown_remaining = max(0, state.global_cooldown_until - now)
                active_cam = state.active_orientation_camera
                current_job_plate = state.current_job.plate if state.current_job else 'None'
            logger.info(f"[{cam_name}] STATUS: frames={frame_counter}, front_consecutive={front_consecutive}, "
                       f"active_camera={active_cam}, current_job={current_job_plate}, "
                       f"global_cooldown={cooldown_remaining:.1f}s")
            last_status_log = now

        # Global suppression
        with state.active_camera_lock:
            if state.active_orientation_camera is not None and state.active_orientation_camera != cam_name:
                time.sleep(0.2)
                continue
            if now < state.global_cooldown_until:
                if frame_counter % 100 == 0:
                    logger.debug(f"[{cam_name}] Global cooldown active: {state.global_cooldown_until - now:.1f}s remaining")
                time.sleep(0.2)
                continue

        # Read frame
        if now - last_frame_time < ANPR_FRAME_TIME:
            time.sleep(0.05)
            continue
        last_frame_time = now

        ret, frame = state.streams[cam_name].read()
        if not ret:
            logger.warning(f"[{cam_name}] Failed to read frame")
            time.sleep(0.1)
            continue

        # Orientation classification
        orientation_frame = cv2.resize(frame, (224, 224))
        results = state.orientation_model(orientation_frame, verbose=False)

        is_front = False
        is_rear = False
        confidence = 0.0
        class_name = None
        MIN_FRONT_CONFIDENCE = 0.6

        if results is not None and len(results) > 0:
            if hasattr(results[0], 'probs') and results[0].probs is not None:
                top1_idx = results[0].probs.top1
                top1_conf = results[0].probs.top1conf.item()
                class_name = state.orientation_model.names[top1_idx]
                if class_name.upper() == "FRONT" and top1_conf >= MIN_FRONT_CONFIDENCE:
                    is_front = True
                    confidence = top1_conf
                    orientation_log_counter += 1
                    if orientation_log_counter % 5 == 0:
                        logger.info(f"[{cam_name}] 🔍 ORIENTATION: FRONT (conf={confidence:.3f}) - consecutive: {front_consecutive+1}")
                        last_orientation_result = "FRONT"
                elif class_name.upper() == "REAR":
                    is_rear = True
                    if last_orientation_result != "REAR" and frame_counter % 50 == 0:
                        logger.debug(f"[{cam_name}] 🔍 ORIENTATION: REAR (conf={top1_conf:.3f})")
                        last_orientation_result = "REAR"
                else:
                    if last_orientation_result != class_name and frame_counter % 50 == 0:
                        logger.debug(f"[{cam_name}] 🔍 ORIENTATION: {class_name} (conf={top1_conf:.3f}) - ignoring")
                        last_orientation_result = class_name
            else:
                if results[0].boxes is not None and len(results[0].boxes) > 0:
                    boxes = results[0].boxes
                    cls_id = int(boxes.cls[0].cpu().numpy())
                    class_name = state.orientation_model.names[cls_id]
                    confidence = float(boxes.conf[0].cpu().numpy())
                    if class_name.upper() == "FRONT" and confidence >= MIN_FRONT_CONFIDENCE:
                        is_front = True
                        orientation_log_counter += 1
                        if orientation_log_counter % 5 == 0:
                            logger.info(f"[{cam_name}] 🔍 DETECTION: FRONT (conf={confidence:.3f}) - consecutive: {front_consecutive+1}")
                    elif class_name.upper() == "REAR":
                        is_rear = True
                        if last_orientation_result != "REAR" and frame_counter % 50 == 0:
                            logger.debug(f"[{cam_name}] 🔍 DETECTION: REAR (conf={confidence:.3f})")
                            last_orientation_result = "REAR"
                    else:
                        if last_orientation_result != class_name and frame_counter % 50 == 0:
                            logger.debug(f"[{cam_name}] 🔍 DETECTION: {class_name} (conf={confidence:.3f}) - ignoring")
                            last_orientation_result = class_name
                else:
                    if last_orientation_result != "NO_DETECTION" and frame_counter % 100 == 0:
                        logger.debug(f"[{cam_name}] 🔍 No valid output from orientation model")
                        last_orientation_result = "NO_DETECTION"
        else:
            if last_orientation_result != "NO_DETECTION" and frame_counter % 100 == 0:
                logger.debug(f"[{cam_name}] 🔍 No results from orientation model")
                last_orientation_result = "NO_DETECTION"

        with state.active_camera_lock:
            am_i_active = (state.active_orientation_camera == cam_name)

        # ─── REAR DETECTION – create fallback if we have a plate ───
        if am_i_active and is_rear:
            if plate_counter:
                best_plate = max(plate_counter.items(), key=lambda x: x[1])[0]
                logger.info(f"[{cam_name}] 🔄 REAR detected – creating fallback job with best plate '{best_plate}' (count={plate_counter[best_plate]})")

                # ─── CACHE CHECK BEFORE CREATING FALLBACK JOB ──────────
                with recent_plates_lock:
                    # Clean expired entries
                    expired = [p for p, (ts, _) in recent_plates.items() if now - ts > RECENT_PLATE_WINDOW_SEC]
                    for p in expired:
                        del recent_plates[p]

                    record_type = CAM_RECORD_TYPE[cam_name]
                    skip = False
                    for existing_plate, (ts, existing_type) in recent_plates.items():
                        if existing_type != record_type:
                            continue
                        if plate_similarity(best_plate, existing_plate) >= RECENT_PLATE_SIMILARITY_THRESHOLD:
                            logger.info(f"[{cam_name}] ⏸️ Skipping fallback job for '{best_plate}' – similar to recently processed '{existing_plate}' (sim={plate_similarity(best_plate, existing_plate):.2f})")
                            skip = True
                            break

                    if skip:
                        with state.active_camera_lock:
                            state.active_orientation_camera = None
                        plate_counter.clear()
                        stabilized_plate = None
                        front_consecutive = 0
                        fallback_created = False
                        continue

                    # Add to cache
                    recent_plates[best_plate] = (now, record_type)

                # ─── CREATE FALLBACK JOB ──────────────────────────────
                fallback_created = True
                ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                anpr_path = f"anpr_images/{best_plate}_{ts}.jpg"
                plate_crop_path = f"plates/{best_plate}_{ts}.jpg"
                cv2.imwrite(anpr_path, frame, [cv2.IMWRITE_JPEG_QUALITY, SAVE_IMAGE_QUALITY])
                if 'plate_crop' in locals() and plate_crop.size > 0:
                    cv2.imwrite(plate_crop_path, plate_crop, [cv2.IMWRITE_JPEG_QUALITY, SAVE_IMAGE_QUALITY])
                else:
                    plate_crop_path = ""

                with state.job_lock:
                    if state.current_job is None:
                        state.current_job = state.Job(
                            plate=best_plate,
                            cam=cam_name,
                            record_type=CAM_RECORD_TYPE[cam_name],
                            anpr_path=anpr_path,
                            started_at=time.time(),
                            use_leftmost=CAM_USE_LEFTMOST[cam_name]
                        )
                        logger.info("=" * 60)
                        logger.info(f"[{cam_name}] ✅ FALLBACK JOB CREATED (REAR) for plate: {best_plate}")
                        logger.info(f"[{cam_name}] Record type: {CAM_RECORD_TYPE[cam_name]}")
                        logger.info("=" * 60)
                    else:
                        logger.error(f"[{cam_name}] ❌ Job collision during rear fallback! Current job: {state.current_job.plate}")
                        with state.active_camera_lock:
                            state.active_orientation_camera = None
                        continue

                with state.last_processed_lock:
                    state.last_processed_plate = best_plate
                    state.last_processed_time = time.time()
                with state.active_camera_lock:
                    state.active_orientation_camera = None
                    state.global_cooldown_until = time.time() + GLOBAL_COOLDOWN_SEC

                plate_counter.clear()
                stabilized_plate = None
                front_consecutive = 0
                plate_change_count = 0
                fallback_created = False
                continue
            else:
                logger.info(f"[{cam_name}] 🔄 REAR detected while ACTIVE - vehicle has passed, releasing camera")
                with state.active_camera_lock:
                    state.active_orientation_camera = None
                front_consecutive = 0
                plate_counter.clear()
                stabilized_plate = None
                plate_change_count = 0
                fallback_created = False
                time.sleep(0.05)
                continue

        # Update front consecutive (only if not active)
        if not am_i_active:
            if is_front:
                front_consecutive += 1
                if front_consecutive == front_stable_frames - 5:
                    logger.info(f"[{cam_name}] 🔥 FRONT detected for {front_consecutive} frames, need {front_stable_frames} to win")
                elif front_consecutive == front_stable_frames:
                    logger.info(f"[{cam_name}] 🎯 FRONT threshold reached! ({front_consecutive}/{front_stable_frames})")
            else:
                if front_consecutive > 0 and front_consecutive < front_stable_frames:
                    logger.debug(f"[{cam_name}] FRONT streak broken at {front_consecutive} frames")
                front_consecutive = 0

        # Quick duplicate check before winning race
        if not am_i_active and front_consecutive >= front_stable_frames:
            plate_frame = cv2.resize(frame, (640, 640))
            plate_results = state.plate_model(plate_frame, conf=plate_conf, verbose=False)
            if plate_results is not None and len(plate_results) > 0:
                if plate_results[0].boxes is not None and len(plate_results[0].boxes) > 0:
                    boxes = plate_results[0].boxes.xyxy.cpu().numpy()
                    if len(boxes) == 1:
                        h, w = frame.shape[:2]
                        x1, y1, x2, y2 = map(int, boxes[0])
                        x1 = int(x1 * w / 640)
                        x2 = int(x2 * w / 640)
                        y1 = int(y1 * h / 640)
                        y2 = int(y2 * h / 640)
                        plate_crop = frame[y1:y2, x1:x2]
                        if plate_crop.size > 0:
                            try:
                                _, img_encoded = cv2.imencode('.jpg', plate_crop)
                                files = {"file": ("plate.jpg", img_encoded.tobytes(), "image/jpeg")}
                                response = requests.post(OCR_URL, files=files, timeout=OCR_TIMEOUT)
                                if response.status_code == 200:
                                    data = response.json()
                                    raw_plate = data.get("cleaned_text", "")
                                    if raw_plate:
                                        valid, _ = validate_plate(raw_plate)
                                        if not valid:
                                            corrected = correct_plate(raw_plate)
                                            if corrected:
                                                quick_plate = corrected
                                            else:
                                                quick_plate = None
                                        else:
                                            quick_plate = raw_plate
                                        if quick_plate:
                                            with state.last_processed_lock:
                                                if state.last_processed_plate:
                                                    similarity = plate_similarity(quick_plate, state.last_processed_plate)
                                                    if similarity >= 0.75:
                                                        logger.info(f"[{cam_name}] ⏸️ Quick check: '{quick_plate}' matches last_processed_plate '{state.last_processed_plate}' (similarity={similarity:.2f})")
                                                        logger.info(f"[{cam_name}] Starting global cooldown instead of creating duplicate job")
                                                        with state.active_camera_lock:
                                                            state.global_cooldown_until = time.time() + GLOBAL_COOLDOWN_SEC
                                                        front_consecutive = 0
                                                        continue
                            except Exception as e:
                                logger.debug(f"[{cam_name}] Quick OCR failed: {e}")

        # Try to become active
        if not am_i_active:
            if front_consecutive >= front_stable_frames:
                with state.active_camera_lock:
                    if state.active_orientation_camera is None and now >= state.global_cooldown_until:
                        state.active_orientation_camera = cam_name
                        am_i_active = True
                        plate_counter.clear()
                        stabilized_plate = None
                        plate_reading_started_at = now
                        plate_reading_deadline = now + PLATE_OCR_TIMEOUT
                        last_plate_detection_time = now
                        plate_change_count = 0
                        fallback_created = False
                        logger.info("=" * 60)
                        logger.info(f"[{cam_name}] 🏆 WON FRONT RACE! (stable for {front_stable_frames} frames)")
                        logger.info(f"[{cam_name}] Now the ACTIVE camera for plate reading")
                        logger.info(f"[{cam_name}] Plate reading deadline: {plate_reading_deadline - now:.0f}s")
                        logger.info("=" * 60)
            if not am_i_active:
                continue

        # Active camera – plate detection & OCR
        if am_i_active:
            if now > plate_reading_deadline:
                logger.warning(f"[{cam_name}] ⏰ Plate reading timeout, releasing camera")
                with state.active_camera_lock:
                    state.active_orientation_camera = None
                front_consecutive = 0
                plate_counter.clear()
                stabilized_plate = None
                fallback_created = False
                continue

            if now - last_plate_detection_time > 15:
                logger.warning(f"[{cam_name}] No plate detected for 15s, releasing camera")
                with state.active_camera_lock:
                    state.active_orientation_camera = None
                front_consecutive = 0
                plate_counter.clear()
                stabilized_plate = None
                fallback_created = False
                continue

            if plate_change_count > 20:
                logger.warning(f"[{cam_name}] Too many plate changes ({plate_change_count}), releasing camera")
                with state.active_camera_lock:
                    state.active_orientation_camera = None
                front_consecutive = 0
                plate_counter.clear()
                stabilized_plate = None
                plate_change_count = 0
                fallback_created = False
                continue

        if am_i_active and not plate_counter and frame_counter % 50 == 0:
            logger.info(f"[{cam_name}] 📷 ACTIVE: Scanning for license plate...")

        # Run plate detection
        plate_frame = cv2.resize(frame, (640, 640))
        results = state.plate_model(plate_frame, conf=plate_conf, verbose=False)

        if results is None or len(results) == 0:
            time.sleep(0.05)
            continue
        if results[0].boxes is None or len(results[0].boxes) == 0:
            time.sleep(0.05)
            continue

        boxes = results[0].boxes.xyxy.cpu().numpy()
        if len(boxes) != 1:
            time.sleep(0.05)
            continue

        last_plate_detection_time = now

        if not plate_counter:
            logger.info(f"[{cam_name}] 🎯 Plate detected! Starting OCR stabilization...")

        # Crop plate ROI
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = map(int, boxes[0])
        x1 = int(x1 * w / 640)
        x2 = int(x2 * w / 640)
        y1 = int(y1 * h / 640)
        y2 = int(y2 * h / 640)
        plate_crop = frame[y1:y2, x1:x2]

        if plate_crop.size == 0:
            logger.warning(f"[{cam_name}] Empty plate crop, skipping")
            continue

        # OCR
        try:
            _, img_encoded = cv2.imencode('.jpg', plate_crop)
            files = {"file": ("plate.jpg", img_encoded.tobytes(), "image/jpeg")}
            response = requests.post(OCR_URL, files=files, timeout=OCR_TIMEOUT)
            if response.status_code != 200:
                logger.debug(f"[{cam_name}] OCR HTTP error: {response.status_code}")
                continue
            data = response.json()
            raw_plate = data.get("cleaned_text", "")
            if not raw_plate:
                logger.debug(f"[{cam_name}] No text from OCR")
                continue

            raw_plate = raw_plate.upper().strip()
            cleaned_plate = strip_india_marking(raw_plate)
            logger.debug(f"[{cam_name}] OCR raw='{raw_plate}' cleaned='{cleaned_plate}'")
        except Exception as e:
            logger.debug(f"[{cam_name}] OCR exception: {e}")
            continue

        # Validate & correct
        valid, reason = validate_plate(cleaned_plate)
        if not valid:
            corrected = correct_plate(cleaned_plate)
            if corrected:
                logger.debug(f"[{cam_name}] Corrected '{raw_plate}' → '{corrected}'")
                plate = corrected
                valid, _ = validate_plate(plate)
                if not valid:
                    logger.debug(f"[{cam_name}] Corrected plate '{plate}' still invalid")
                    continue
            else:
                logger.debug(f"[{cam_name}] Rejected '{raw_plate}' - {reason}")
                continue
        else:
            plate = cleaned_plate

        # ─── SPECIAL PLATE ──────────────────────────────────────────────
        if plate in SPECIAL_19KG_PLATES:
            plate_counter[plate] = plate_ocr_stable_frames
            logger.info(f"[{cam_name}] 🎯 SPECIAL PLATE DETECTED: '{plate}' – forcing immediate stabilization")

        # ─── TRUSTED PLATE ──────────────────────────────────────────────
        if plate in TRUSTED_PLATE_COUNTS:
            plate_counter[plate] = plate_ocr_stable_frames
            logger.info(f"[{cam_name}] 🎯 TRUSTED PLATE DETECTED: '{plate}' – forcing immediate stabilization")

        # Frequency-based stabilization
        plate_counter[plate] += 1
        current_count = plate_counter[plate]

        logger.info(f"[{cam_name}] 📝 OCR result: '{plate}' (count={current_count}/{plate_ocr_stable_frames})")

        if current_count % 3 == 0:
            logger.debug(f"[{cam_name}] Plate frequency map: {dict(sorted(plate_counter.items(), key=lambda x: x[1], reverse=True))}")

        if current_count >= plate_ocr_stable_frames and stabilized_plate is None:
            stabilized_plate = plate
            logger.info(f"[{cam_name}] 🎯 Plate stabilized by frequency voting: {plate} ({current_count} detections)")

        # Global deduplication (with exact and similar check)
        with state.last_processed_lock:
            if state.last_processed_plate:
                if plate == state.last_processed_plate:
                    if (now - state.last_processed_time) < GLOBAL_COOLDOWN_SEC:
                        logger.info(f"[{cam_name}] ⏸️ Skipping exact duplicate '{plate}' within {GLOBAL_COOLDOWN_SEC}s cooldown")
                        with state.active_camera_lock:
                            state.active_orientation_camera = None
                            state.global_cooldown_until = time.time() + GLOBAL_COOLDOWN_SEC
                        plate_counter.clear()
                        stabilized_plate = None
                        front_consecutive = 0
                        fallback_created = False
                        continue
                else:
                    similarity = plate_similarity(plate, state.last_processed_plate)
                    if similarity >= 0.75:
                        logger.info(f"[{cam_name}] 🔄 Similar plate detected: '{plate}' ≈ '{state.last_processed_plate}' (similarity={similarity:.2f}) → treating as SAME vehicle")
                        with state.active_camera_lock:
                            state.global_cooldown_until = time.time() + GLOBAL_COOLDOWN_SEC
                            state.active_orientation_camera = None
                        plate_counter.clear()
                        stabilized_plate = None
                        front_consecutive = 0
                        fallback_created = False
                        continue

        # ─── If stabilized, create job ────────────────────────────────
        if stabilized_plate is not None:
            plate = stabilized_plate
            logger.info("=" * 60)
            logger.info(f"[{cam_name}] 🎉 PLATE STABILIZED! count={plate_counter[plate]}")
            logger.info(f"[{cam_name}] Final plate: '{plate}'")
            logger.info("=" * 60)

            # ─── RECENT CACHE DEDUPLICATION ──────────────────────────────
            with recent_plates_lock:
                # Clean expired entries
                expired = [p for p, (ts, _) in recent_plates.items() if now - ts > RECENT_PLATE_WINDOW_SEC]
                for p in expired:
                    del recent_plates[p]

                record_type = CAM_RECORD_TYPE[cam_name]
                skip = False
                for existing_plate, (ts, existing_type) in recent_plates.items():
                    if existing_type != record_type:
                        continue
                    if plate_similarity(plate, existing_plate) >= RECENT_PLATE_SIMILARITY_THRESHOLD:
                        logger.info(f"[{cam_name}] ⏸️ Skipping job for '{plate}' – similar to recently processed '{existing_plate}' (sim={plate_similarity(plate, existing_plate):.2f})")
                        skip = True
                        break

                if skip:
                    with state.active_camera_lock:
                        state.active_orientation_camera = None
                    plate_counter.clear()
                    stabilized_plate = None
                    front_consecutive = 0
                    fallback_created = False
                    continue

                # Add to cache (even if job creation later fails, we keep it)
                recent_plates[plate] = (now, record_type)

            # ─── PER‑CAMERA COOLDOWN ──────────────────────────────────────
            cam_history = state.last_captured_plate[cam_name]
            last_time = cam_history.get(plate, 0)
            if (now - last_time) < cooldown_sec:
                logger.warning(f"[{cam_name}] ⏸️ Plate '{plate}' still in per‑camera cooldown ({cooldown_sec}s), resetting")
                with state.active_camera_lock:
                    state.active_orientation_camera = None
                plate_counter.clear()
                stabilized_plate = None
                front_consecutive = 0
                fallback_created = False
                continue

            # ─── IN_PROGRESS CHECK ────────────────────────────────────────
            with state.IN_PROGRESS_LOCK:
                if plate in state.IN_PROGRESS:
                    logger.warning(f"[{cam_name}] ⚠️ Plate '{plate}' already in progress, skipping")
                    with state.active_camera_lock:
                        state.active_orientation_camera = None
                    plate_counter.clear()
                    stabilized_plate = None
                    front_consecutive = 0
                    fallback_created = False
                    continue
                state.IN_PROGRESS.add(plate)
                logger.info(f"[{cam_name}] Added '{plate}' to IN_PROGRESS set")

            # ─── SAVE EVIDENCE ────────────────────────────────────────────
            ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            anpr_path = f"anpr_images/{plate}_{ts}.jpg"
            plate_crop_path = f"plates/{plate}_{ts}.jpg"
            cv2.imwrite(anpr_path, frame, [cv2.IMWRITE_JPEG_QUALITY, SAVE_IMAGE_QUALITY])
            cv2.imwrite(plate_crop_path, plate_crop, [cv2.IMWRITE_JPEG_QUALITY, SAVE_IMAGE_QUALITY])
            logger.info(f"[{cam_name}] 💾 Saved ANPR image: {anpr_path}")
            logger.info(f"[{cam_name}] 💾 Saved plate crop: {plate_crop_path}")

            # ─── CREATE JOB ────────────────────────────────────────────────
            with state.job_lock:
                if state.current_job is None:
                    state.current_job = state.Job(
                        plate=plate,
                        cam=cam_name,
                        record_type=CAM_RECORD_TYPE[cam_name],
                        anpr_path=anpr_path,
                        started_at=time.time(),
                        use_leftmost=CAM_USE_LEFTMOST[cam_name]
                    )
                    logger.info("=" * 60)
                    logger.info(f"[{cam_name}] ✅ JOB CREATED for plate: {plate}")
                    logger.info(f"[{cam_name}] Record type: {CAM_RECORD_TYPE[cam_name]}")
                    logger.info(f"[{cam_name}] Use leftmost: {CAM_USE_LEFTMOST[cam_name]}")
                    logger.info("=" * 60)
                else:
                    logger.error(f"[{cam_name}] ❌ Job collision! Current job: {state.current_job.plate}, incoming: {plate}")
                    with state.IN_PROGRESS_LOCK:
                        state.IN_PROGRESS.discard(plate)
                    with state.active_camera_lock:
                        state.active_orientation_camera = None
                    plate_counter.clear()
                    stabilized_plate = None
                    front_consecutive = 0
                    fallback_created = False
                    continue

            # ─── UPDATE HISTORY ────────────────────────────────────────────
            cam_history[plate] = now
            with state.last_processed_lock:
                state.last_processed_plate = plate
                state.last_processed_time = time.time()
                plate_counter.clear()
                stabilized_plate = None
                logger.debug(f"[{cam_name}] Updated last_processed_plate to '{plate}'")

            with state.active_camera_lock:
                state.active_orientation_camera = None
                state.global_cooldown_until = time.time() + GLOBAL_COOLDOWN_SEC
                logger.info(f"[{cam_name}] Released active camera, global cooldown: {GLOBAL_COOLDOWN_SEC}s")

            plate_counter.clear()
            stabilized_plate = None
            front_consecutive = 0
            plate_change_count = 0
            fallback_created = False
            logger.info(f"[{cam_name}] Local state reset, waiting for next vehicle")

        # ─── FALLBACK (entry only) ──────────────────────────────────────
        if cam_name == "cam5" and not fallback_created and am_i_active:
            elapsed_since_active = now - plate_reading_started_at
            if elapsed_since_active > fallback_timeout:
                special_detected = any(p in SPECIAL_19KG_PLATES for p in plate_counter.keys())
                if special_detected:
                    logger.info(f"[{cam_name}] ⏸️ Special plate detected in counter, skipping fallback and waiting for stabilization")
                    time.sleep(0.05)
                    continue

                if plate_counter:
                    best_plate = max(plate_counter.items(), key=lambda x: x[1])[0]
                    fallback_plate = best_plate
                    logger.info(f"[{cam_name}] Using best detected plate '{fallback_plate}' (count={plate_counter[fallback_plate]}) for fallback job")
                else:
                    fallback_plate = f"ENTRY_{int(now)}_{frame_counter}"
                    logger.warning(f"[{cam_name}] ⏰ No valid plate stabilised after {elapsed_since_active:.1f}s, creating fallback job with plate '{fallback_plate}'")

                # ─── CACHE CHECK FOR FALLBACK ────────────────────────────
                with recent_plates_lock:
                    # Clean expired
                    expired = [p for p, (ts, _) in recent_plates.items() if now - ts > RECENT_PLATE_WINDOW_SEC]
                    for p in expired:
                        del recent_plates[p]

                    record_type = CAM_RECORD_TYPE[cam_name]
                    skip = False
                    for existing_plate, (ts, existing_type) in recent_plates.items():
                        if existing_type != record_type:
                            continue
                        if plate_similarity(fallback_plate, existing_plate) >= RECENT_PLATE_SIMILARITY_THRESHOLD:
                            logger.info(f"[{cam_name}] ⏸️ Skipping fallback job for '{fallback_plate}' – similar to recently processed '{existing_plate}' (sim={plate_similarity(fallback_plate, existing_plate):.2f})")
                            skip = True
                            break

                    if skip:
                        with state.active_camera_lock:
                            state.active_orientation_camera = None
                        plate_counter.clear()
                        stabilized_plate = None
                        front_consecutive = 0
                        fallback_created = False
                        continue

                    # Add to cache
                    recent_plates[fallback_plate] = (now, record_type)

                # ─── CREATE FALLBACK JOB ──────────────────────────────────
                fallback_created = True
                ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                anpr_path = f"anpr_images/{fallback_plate}_{ts}.jpg"
                plate_crop_path = f"plates/{fallback_plate}_{ts}.jpg"
                cv2.imwrite(anpr_path, frame, [cv2.IMWRITE_JPEG_QUALITY, SAVE_IMAGE_QUALITY])
                if plate_crop.size > 0:
                    cv2.imwrite(plate_crop_path, plate_crop, [cv2.IMWRITE_JPEG_QUALITY, SAVE_IMAGE_QUALITY])
                else:
                    plate_crop_path = ""

                with state.job_lock:
                    if state.current_job is None:
                        state.current_job = state.Job(
                            plate=fallback_plate,
                            cam=cam_name,
                            record_type=CAM_RECORD_TYPE[cam_name],
                            anpr_path=anpr_path,
                            started_at=time.time(),
                            use_leftmost=CAM_USE_LEFTMOST[cam_name]
                        )
                        logger.info("=" * 60)
                        logger.info(f"[{cam_name}] ✅ FALLBACK JOB CREATED for plate: {fallback_plate}")
                        logger.info(f"[{cam_name}] Record type: {CAM_RECORD_TYPE[cam_name]}")
                        logger.info("=" * 60)
                    else:
                        logger.error(f"[{cam_name}] ❌ Job collision during fallback! Current job: {state.current_job.plate}")
                        with state.active_camera_lock:
                            state.active_orientation_camera = None
                        continue

                with state.last_processed_lock:
                    state.last_processed_plate = fallback_plate
                    state.last_processed_time = time.time()
                with state.active_camera_lock:
                    state.active_orientation_camera = None
                    state.global_cooldown_until = time.time() + GLOBAL_COOLDOWN_SEC

                plate_counter.clear()
                stabilized_plate = None
                front_consecutive = 0
                plate_change_count = 0
                fallback_created = False
                continue

        time.sleep(0.01)