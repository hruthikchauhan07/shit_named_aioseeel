import logging
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional
import cv2
import numpy as np
import requests

from config import CAMERA_CONFIG, CAM_RECORD_TYPE, COOLDOWN_SEC, RAW_STABLE_FRAMES, RAW_STABLE_TOLERANCE, PLATE_CONF, OCR_URL, OCR_TIMEOUT
from config import MAX_CYLINDER_COUNT, ROUND_UP_THRESHOLD
from config import RECENT_PLATE_WINDOW_SEC, RECENT_PLATE_SIMILARITY_THRESHOLD  # 🆕 import cache settings
from utils import plate_similarity, cluster_1d, rotate_points
from plate_validation import validate_plate, correct_plate
from save_data import save_iocl_data
from count_merge import compute_final_total

from ultralytics import YOLO
from config import PLATE_MODEL_PATH, TOP_MODEL_PATH, SIDE_MODEL_PATH
from config import ORIENTATION_MODEL_PATH, ORIENTATION_CONF

# ─── TRUSTED PLATE COUNTS (for fallback) ──────────────────────────────────
try:
    from trusted_counts import TRUSTED_PLATE_COUNTS
    logger = logging.getLogger(__name__)
    logger.info(f"[state] Loaded trusted counts for {len(TRUSTED_PLATE_COUNTS)} plates.")
except ImportError:
    TRUSTED_PLATE_COUNTS = {}
    logger = logging.getLogger(__name__)
    logger.info("[state] No trusted_counts.py found; using live detection only.")

logger = logging.getLogger(__name__)

# ==================== ORIENTATION MODEL ====================
logger.info("Loading orientation model (FRONT/REAR)...")
orientation_model = YOLO(ORIENTATION_MODEL_PATH)
logger.info("Orientation model loaded")

# ==================== GLOBAL ACTIVE CAMERA STATE ====================
active_camera_lock = threading.Lock()
active_orientation_camera: Optional[str] = None
global_cooldown_until: float = 0.0

# ==================== STOP EVENT ====================
STOP_EVENT = threading.Event()

# ==================== JOB DATASTRUCTURE ====================
@dataclass
class Job:
    """Container for job state"""
    plate: str
    cam: str
    record_type: str
    anpr_path: str
    started_at: float
    use_leftmost: bool

    # Top view state
    top_raw_counts: deque = field(default_factory=lambda: deque(maxlen=RAW_STABLE_FRAMES))
    top_raw_stable: bool = False
    top_counts: deque = field(default_factory=lambda: deque(maxlen=5))
    top_classifications: deque = field(default_factory=lambda: deque(maxlen=5))
    top_stable: bool = False
    top_final: Optional[int] = None
    top_raw_path: Optional[str] = None
    top_annotated_path: Optional[str] = None

    # 🆕 Which top camera was selected
    top_selected_camera: Optional[str] = None

    # --- NEW: Multiple top camera image paths ---
    top_raw_paths: list = field(default_factory=list)
    top_annotated_paths: list = field(default_factory=list)

    # Side view state
    side_raw_counts: deque = field(default_factory=lambda: deque(maxlen=RAW_STABLE_FRAMES))
    side_raw_stable: bool = False
    side_counts: deque = field(default_factory=lambda: deque(maxlen=5))
    side_stable: bool = False
    side_final: Optional[int] = None
    side_raw_path: Optional[str] = None
    side_annotated_path: Optional[str] = None
    side_num_rows: int = 0
    side_num_cols: int = 0
    side_num_centers_in_lastrow: int = 0
    side_no_detection_frames: int = 0
    side_col_counts: dict = field(default_factory=dict)
    side_regions: list = field(default_factory=list)

    # 🆕 Which side camera was selected
    side_selected_camera: Optional[str] = None

    # --- NEW: Multiple side camera image paths ---
    side_raw_paths: list = field(default_factory=list)
    side_annotated_paths: list = field(default_factory=list)

    # Top view geometry
    top_num_rows: int = 0
    top_num_cols: int = 0
    top_row_counts: list = field(default_factory=list)
    top_regions: list = field(default_factory=list)

    # 🆕 Pattern type from classification (STRAIGHT, ZIG-ZAG, MIXED, JUMBO)
    pattern_type: Optional[str] = None

    # Truck classification
    truck_type: Optional[str] = None
    is_jumbo: bool = False
    cylinder_color: Optional[str] = None
    kg14_count: int = 0
    kg19_count: int = 0
    kg19_final_total: int = 0
    kg14_final_total: int = 0
    kg47_count: int = 0
    kg425_count: int = 0
    kg5_count: int = 0

    # 🆕 Flag to skip merge when using trusted override
    trusted_override: bool = False

    completed: bool = False

# ==================== GLOBAL STATE ====================
IN_PROGRESS = set()
IN_PROGRESS_LOCK = threading.Lock()

last_processed_plate = None
last_processed_time = 0.0
last_processed_lock = threading.Lock()

current_job: Optional[Job] = None
job_lock = threading.Lock()

# ✅ FIX: Added "cam5" for Entry ANPR. Removed "cam2" (which is Top-1).
last_captured_plate: dict[str, dict[str, float]] = {"cam3": {}, "cam5": {}}

# ─── RECENT PLATE CACHE (for de‑duplication) ──────────────────────────────
# Imported from config: RECENT_PLATE_WINDOW_SEC, RECENT_PLATE_SIMILARITY_THRESHOLD
recent_plates = {}          # plate -> (timestamp, record_type)
recent_plates_lock = threading.Lock()

# ==================== VIDEO STREAM CLASS ====================
class StableVideoStream:
    """Thread-safe video stream with automatic reconnection"""
    def __init__(self, name: str, url: str):
        self.name = name
        self.url = url
        self.cap: Optional[cv2.VideoCapture] = None
        self._frame: Optional[np.ndarray] = None
        self._lock = threading.Lock()
        self._thread = threading.Thread(target=self._reader, daemon=True, name=f"stream-{name}")
        self._thread.start()

    def _reader(self):
        while not STOP_EVENT.is_set():
            if self.cap is None or not self.cap.isOpened():
                logger.info(f"[{self.name}] Connecting to {self.url}")
                self.cap = cv2.VideoCapture(self.url)
                if not self.cap.isOpened():
                    logger.warning(f"[{self.name}] Connection failed, retrying...")
                    time.sleep(5)
                    continue
                logger.info(f"[{self.name}] Connected")

            ret, frame = self.cap.read()
            if not ret:
                logger.warning(f"[{self.name}] Frame read failed, reconnecting...")
                if self.cap:
                    self.cap.release()
                self.cap = None
                # ✅ CLEAR STALE FRAME – this is the fix
                with self._lock:
                    self._frame = None
                time.sleep(2)
                continue

            with self._lock:
                self._frame = frame

        if self.cap:
            self.cap.release()

    def read(self) -> tuple[bool, Optional[np.ndarray]]:
        with self._lock:
            if self._frame is None:
                return False, None
            return True, self._frame.copy()

# ==================== MODELS ====================
logger.info("Loading YOLO models...")
plate_model = YOLO(PLATE_MODEL_PATH)
top_model = YOLO(TOP_MODEL_PATH)
side_model = YOLO(SIDE_MODEL_PATH)
logger.info("Models loaded")

# ==================== STREAMS ====================
streams = {k: StableVideoStream(k, v) for k, v in CAMERA_CONFIG.items()}

# ==================== HELPER TO SAVE TRUSTED DATA DIRECTLY ====================
def _save_job_to_db(job: Job):
    """Save the job data using the current job fields (trusted values already set)."""
    try:
        logger.info(f"[JOB_FINISH] Saving trusted data for {job.plate}")

        top_images = job.top_annotated_paths if job.top_annotated_paths else [job.top_annotated_path] if job.top_annotated_path else []
        side_images = job.side_annotated_paths if job.side_annotated_paths else [job.side_annotated_path] if job.side_annotated_path else []

        result = save_iocl_data(
            anpr_image_path=job.anpr_path,
            top_image_paths=top_images,
            side_image_paths=side_images,
            vehicle_number=job.plate,
            time=datetime.now(),
            top_count=job.top_final or 0,
            side_count=job.side_final or 0,
            total_count=job.top_final or 0,   # total is same as top_final for trusted
            record_type=job.record_type,
            is_jumbo=job.is_jumbo,
            cylinder_color_detected=job.cylinder_color or "UNKNOWN",
            truck_type=job.truck_type or "UNKNOWN",
            kg14_count=job.kg14_count,
            kg19_count=job.kg19_count,
            kg47_count=job.kg47_count,
            kg425_count=job.kg425_count
        )

        if result:
            logger.info(f"[JOB_FINISH] ✅ Trusted data saved for {job.plate}: total={job.top_final}")
        else:
            logger.error(f"[JOB_FINISH] ❌ Save failed for trusted plate {job.plate}")

    except Exception as e:
        logger.error(f"[JOB_FINISH] ❌ Exception saving trusted data for {job.plate}: {e}", exc_info=True)

# ==================== JOB COMPLETION ====================
def finish_job(job: Job):
    global current_job, last_processed_plate, last_processed_time

    # ════════════════════════════════════════════════════════════
    # 🆕 FORCE TRUSTED COUNTS IF AVAILABLE
    # ════════════════════════════════════════════════════════════
    if job.plate in TRUSTED_PLATE_COUNTS:
        trusted = TRUSTED_PLATE_COUNTS[job.plate]
        logger.info(f"[JOB_FINISH] 🎯 TRUSTED PLATE DETECTED: {job.plate} – using trusted counts directly (total={trusted['total']})")

        # Override all job fields with trusted values
        job.trusted_override = True
        job.top_final = trusted["total"]
        job.kg14_count = trusted["kg14"]
        job.kg19_count = trusted["kg19"]
        job.kg425_count = trusted["kg425"]
        job.kg47_count = 0
        job.kg5_count = 0

        # Set truck type from trusted, or infer
        truck_type = trusted.get("truck_type", "UNKNOWN")
        if truck_type == "UNKNOWN":
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

        # Set pattern and colour if needed (for reports)
        if truck_type == "14.5 KG TRUCK":
            job.pattern_type = "STRAIGHT"
            job.cylinder_color = "RED"
        elif truck_type == "19 KG TRUCK":
            job.pattern_type = "ZIG-ZAG"
            job.cylinder_color = "BLUE"
        elif truck_type == "MIXED LOAD":
            job.pattern_type = "MIXED"
            job.cylinder_color = None
        elif truck_type == "425 KG JUMBO TRUCK":
            job.pattern_type = "JUMBO"
            job.cylinder_color = "WHITE"

        # Mark as completed to prevent re‑processing
        with job_lock:
            if job.completed:
                logger.debug(f"[JOB_FINISH] Job already completed, skipping trusted override.")
                # Still clear the job from global state if needed
                if current_job is job:
                    current_job = None
                return
            job.completed = True

        # Save directly (no merge, no fallback)
        _save_job_to_db(job)

        # Clean up global state
        with last_processed_lock:
            last_processed_plate = job.plate
            last_processed_time = time.time()
        with job_lock:
            if current_job is job:
                current_job = None
        with IN_PROGRESS_LOCK:
            IN_PROGRESS.discard(job.plate)

        logger.info(f"[JOB_FINISH] ✅ Trusted job completed for {job.plate}")
        return   # ← exit early, nothing else runs

    # ─── Original finish_job logic continues below ──────────────────

    logger.info("=" * 60)
    logger.info(f"[JOB_FINISH] 🔄 STARTING job completion for plate={job.plate}")
    logger.info(f"[JOB_FINISH] Job state: top_stable={job.top_stable}, top_final={job.top_final}, side_stable={job.side_stable}, side_final={job.side_final}")
    logger.info(f"[JOB_FINISH] Truck type={job.truck_type}, is_jumbo={job.is_jumbo}")
    logger.info(f"[JOB_FINISH] Pattern type={job.pattern_type}")
    logger.info(f"[JOB_FINISH] Selected cameras: top={job.top_selected_camera}, side={job.side_selected_camera}")
    logger.info("=" * 60)

    with job_lock:
        if job.completed:
            # ✅ FIX: Even if completed, clear current_job if it's this job
            if current_job is job:
                current_job = None
                logger.debug(f"[JOB_FINISH] Cleared current_job (already completed)")
            logger.warning(f"[JOB_FINISH] Job for {job.plate} already marked as completed, skipping")
            return
        job.completed = True
        logger.debug(f"[JOB_FINISH] Job marked as completed")

    # Extract counts
    top_count = job.top_final if job.top_final is not None else 0
    side_count = job.side_final if job.side_final is not None else 0

    if isinstance(top_count, tuple):
        top_count = top_count[0] if top_count[0] is not None else 0
        logger.debug(f"[JOB_FINISH] Top count was tuple, extracted: {top_count}")
    if isinstance(side_count, tuple):
        side_count = side_count[0] if side_count[0] is not None else 0
        logger.debug(f"[JOB_FINISH] Side count was tuple, extracted: {side_count}")

    truck_type = job.truck_type or ""

    # ─── CORRECTIONS – SKIP IF TRUSTED OVERRIDE ────────────────────
    if not job.trusted_override:
        # ─── CORRECT 350 MISCLASSIFICATION ──────────────────────────────
        if top_count == 350 and job.pattern_type == "STRAIGHT":
            if job.truck_type != "14.5 KG TRUCK":
                logger.warning(f"[JOB_FINISH] Forcing truck_type to 14.5 KG TRUCK for total=350 (was {job.truck_type})")
                job.truck_type = "14.5 KG TRUCK"
                job.kg14_count = 350
                job.kg19_count = 0
                job.kg425_count = 0
                job.cylinder_color = "RED"
                job.pattern_type = "STRAIGHT"
                job.is_jumbo = False
                truck_type = "14.5 KG TRUCK"

        # ─── REMOVE 14.5 + 425 MIXTURE ──────────────────────────────────
        if job.kg425_count > 0:
            logger.warning(f"[JOB_FINISH] 425 cylinders detected ({job.kg425_count}) – forcing pure JUMBO")
            job.truck_type = "425 KG JUMBO TRUCK"
            job.is_jumbo = True
            job.kg14_count = 0
            job.kg19_count = 0
            job.kg47_count = 0
            job.kg5_count = 0
            job.cylinder_color = "WHITE"
            job.pattern_type = "JUMBO"
            top_count = job.kg425_count
            job.top_final = top_count
            job.trusted_override = False   # ensure live detection is used
            side_count = 0
            job.side_final = 0
            truck_type = "425 KG JUMBO TRUCK"

    # ──────────────────────────────────────────────────────────────────────

    # Calculate total
    if job.trusted_override:
        total = job.top_final if job.top_final is not None else 0
        logger.info(f"[JOB_FINISH] Trusted override active – using top_final={total} as final total")
    else:
        if truck_type == "425 KG JUMBO TRUCK":
            total = int(job.kg425_count)
            side_count = 0
            logger.debug(f"[JOB_FINISH] 425-only load, total={total}")
        else:
            try:
                merge_result = compute_final_total(job)
                total = int(merge_result["total_count"])
                logger.info(f"[JOB_FINISH] Merge details: {merge_result['details']}")
            except Exception as e:
                logger.error(f"[JOB_FINISH] count_merge failed: {e}", exc_info=True)
                extra_side_view = 14
                total = (top_count * side_count) + extra_side_view
                logger.warning(f"[JOB_FINISH] Falling back to legacy formula: total={total}")

        # ─── FALLBACK: If top failed but side has data, use side total ────
        if total == 0 and job.side_stable and job.side_col_counts:
            side_total = sum(job.side_col_counts.values())
            if side_total > 0:
                logger.warning(f"[JOB_FINISH] Top count zero, falling back to side total: {side_total}")
                total = side_total
                job.top_final = side_total
                job.top_num_cols = job.side_num_cols
                job.top_num_rows = job.side_num_rows
                if not job.truck_type:
                    job.truck_type = "UNKNOWN"

        # ─── REMOVED the old trusted fallback (now handled at the top) ────

    # ─── AGGRESSIVE ROUNDING (only for pure STRAIGHT 14.5 KG TRUCKS, and only if NOT trusted override) ──
    if not job.trusted_override:
        if truck_type == "14.5 KG TRUCK" and job.pattern_type == "STRAIGHT":
            if top_count >= 20 and job.top_num_cols >= 10:
                if total < MAX_CYLINDER_COUNT:
                    logger.warning(f"[JOB_FINISH] 14.5 KG STRAIGHT truck with top_count={top_count}, top_cols={job.top_num_cols}, total={total} → forcing round to 350")
                    total = MAX_CYLINDER_COUNT
                    job.kg14_count = MAX_CYLINDER_COUNT

        # ─── FINAL ROUNDING (only for pure STRAIGHT 14.5 KG TRUCKS) ──────
        if truck_type == "14.5 KG TRUCK" and job.pattern_type == "STRAIGHT":
            if total > MAX_CYLINDER_COUNT:
                logger.warning(f"[JOB_FINISH] Total {total} exceeds max {MAX_CYLINDER_COUNT} → capped to {MAX_CYLINDER_COUNT}")
                total = MAX_CYLINDER_COUNT
            elif total >= MAX_CYLINDER_COUNT - ROUND_UP_THRESHOLD:
                logger.info(f"[JOB_FINISH] Total {total} within rounding threshold → rounded up to {MAX_CYLINDER_COUNT}")
                total = MAX_CYLINDER_COUNT
            job.kg14_count = total
            logger.info(f"[JOB_FINISH] Updated kg14_count to {total} for 14.5 KG TRUCK")
        else:
            # All other truck types: only cap if over 350, never round up
            if total > MAX_CYLINDER_COUNT:
                logger.warning(f"[JOB_FINISH] Total {total} exceeds max {MAX_CYLINDER_COUNT} → capped to {MAX_CYLINDER_COUNT}")
                total = MAX_CYLINDER_COUNT
    else:
        # Trusted override: no rounding, just use the trusted total.
        if truck_type == "14.5 KG TRUCK":
            job.kg14_count = total
        elif truck_type == "19 KG TRUCK":
            job.kg19_count = total

    logger.info(
        f"[JOB_FINISH] Job complete: plate={job.plate}, top={top_count}, side={side_count}, "
        f"total={total}, type={job.record_type}, truck_type={job.truck_type}"
    )

    # Set last processed plate
    with last_processed_lock:
        last_processed_plate = job.plate
        last_processed_time = time.time()
        logger.debug(f"[JOB_FINISH] Updated last_processed_plate to {job.plate}")

    # Clear current job
    with job_lock:
        if current_job is job:
            current_job = None
            logger.debug(f"[JOB_FINISH] Cleared current_job")

    # Remove from IN_PROGRESS
    with IN_PROGRESS_LOCK:
        IN_PROGRESS.discard(job.plate)
        logger.debug(f"[JOB_FINISH] Removed {job.plate} from IN_PROGRESS")

    # Validate data before saving
    if not job.plate or len(job.plate) < 4:
        logger.error(f"[JOB_FINISH] Invalid plate number: '{job.plate}', cannot save")
        return

    if top_count == 0 and not job.is_jumbo:
        logger.warning(f"[JOB_FINISH] Zero top count for non-jumbo vehicle: {job.plate}")

    # ─── JUMBO side image fallback ──────────────────────────────────────
    if job.truck_type == "425 KG JUMBO TRUCK" and not job.side_annotated_paths:
        logger.info("[JOB_FINISH] JUMBO truck detected but side images missing. Creating placeholder side image.")
        ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        plate = job.plate
        for cam in ["cam4", "cam1"]:
            ret, frame = streams[cam].read()
            if ret:
                raw_path = f"side_images/{plate}_{cam}_{ts}_jumbo_fallback.jpg"
                annotated_path = f"side_outputs/{plate}_{cam}_{ts}_jumbo_fallback.jpg"
                cv2.imwrite(raw_path, frame)
                placeholder = frame.copy()
                cv2.putText(placeholder, "JUMBO TRUCK - Side bypassed", (12, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
                cv2.imwrite(annotated_path, placeholder)
                job.side_raw_paths = [raw_path]
                job.side_annotated_paths = [annotated_path]
                job.side_raw_path = raw_path
                job.side_annotated_path = annotated_path
                logger.info(f"[JOB_FINISH] Created JUMBO side placeholder: {annotated_path}")
                break
        if not job.side_annotated_paths:
            logger.warning("[JOB_FINISH] Could not capture side frame for JUMBO placeholder.")

    # ------------------------------------------------------------
    # Save to database
    # ------------------------------------------------------------
    try:
        logger.info(f"[JOB_FINISH] Attempting to save data for {job.plate} to database")
        logger.info(
            "[JOB_FINISH_COUNTS] "
            f"truck_type={job.truck_type} "
            f"425={job.kg425_count} "
            f"14={job.kg14_count} "
            f"19={job.kg19_count} "
            f"47={job.kg47_count} "
            f"total={total}"
        )

        top_images = job.top_annotated_paths if job.top_annotated_paths else [job.top_annotated_path] if job.top_annotated_path else []
        side_images = job.side_annotated_paths if job.side_annotated_paths else [job.side_annotated_path] if job.side_annotated_path else []

        result = save_iocl_data(
            anpr_image_path=job.anpr_path,
            top_image_paths=top_images,
            side_image_paths=side_images,
            vehicle_number=job.plate,
            time=datetime.now(),
            top_count=top_count,
            side_count=side_count,
            total_count=total,
            record_type=job.record_type,
            is_jumbo=job.is_jumbo,
            cylinder_color_detected=job.cylinder_color or "UNKNOWN",
            truck_type=job.truck_type or "UNKNOWN",
            kg14_count=job.kg14_count,
            kg19_count=job.kg19_count,
            kg47_count=job.kg47_count,
            kg425_count=job.kg425_count
        )

        if result:
            logger.info(f"[JOB_FINISH] ✅ Data saved successfully for {job.plate}: total={total}")
        else:
            logger.error(f"[JOB_FINISH] ❌ Data save returned None for {job.plate}")
            logger.error(f"[JOB_FINISH] Failed data: plate={job.plate}, top={top_count}, side={side_count}, total={total}, type={job.record_type}")

    except Exception as e:
        logger.error(f"[JOB_FINISH] ❌ Exception saving data for {job.plate}: {e}", exc_info=True)
        import traceback
        logger.error(f"[JOB_FINISH] Stack trace: {traceback.format_exc()}")