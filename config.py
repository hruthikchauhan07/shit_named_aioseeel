"""
IOCL Cylinder Counting System - Configuration
INFINOVA NVR C++ PLATFORM
"""

import os
import re

# ════════════════════════════════════════════════════════════════════════════
# ORIENTATION MODEL (FRONT / REAR)
# ════════════════════════════════════════════════════════════════════════════
ORIENTATION_MODEL_PATH = r"C:\Users\admin\Desktop\cyl\approach2\models\front-cls.pt"
ORIENTATION_CONF = 0.7
FRONT_STABLE_FRAMES = 15
PLATE_OCR_STABLE_FRAMES = 9
PLATE_OCR_TIMEOUT = 45
ANPR_FPS = 1
ANPR_FRAME_TIME = 1.0 / ANPR_FPS

# ─── IMAGE COMPRESSION ──────────────────────────────────
SAVE_IMAGE_QUALITY = 70    # 0-100, 70 gives ~60% smaller files with good quality

# ─── RECENT PLATE CACHE (for de‑duplication) ──────────────────────────────
RECENT_PLATE_WINDOW_SEC = 60            # seconds to consider a plate as "recent"
RECENT_PLATE_SIMILARITY_THRESHOLD = 0.80

# ════════════════════════════════════════════════════════════════════════════
# RTSP CAMERA CONFIGURATION – USE RAW PASSWORD (no %40)
# ════════════════════════════════════════════════════════════════════════════
USERNAME = "admin"
PASSWORD_RAW = "Iocl@1234"          # ✅ Raw password for OpenCV
NVR_IP = "192.168.3.250"
PORT = 554

def rtsp_url(channel, subtype=0):
    # ✅ Use raw password – OpenCV handles `@` correctly in the URL
    return f"rtsp://{USERNAME}:{PASSWORD_RAW}@{NVR_IP}:{PORT}/cam/realmonitor?channel={channel}&subtype={subtype}"

CAMERA_CONFIG = {
    "cam1": rtsp_url(1, 0),  # SIDE view 1
    "cam2": rtsp_url(2, 0),  # TOP view 1
    "cam3": rtsp_url(3, 0),  # EXIT ANPR
    "cam4": rtsp_url(4, 0),  # SIDE view 2
    "cam5": rtsp_url(5, 0),  # ENTRY ANPR
    "cam6": rtsp_url(6, 0),  # TOP view 2
}

# ════════════════════════════════════════════════════════════════════════════
# CAMERA ROLES (ANPR only)
# ════════════════════════════════════════════════════════════════════════════
CAM_RECORD_TYPE = {
    "cam3": "exit",   # Channel 3 = Exit
    "cam5": "entry",  # Channel 5 = Entry
}

CAM_USE_LEFTMOST = {
    "cam3": False,    # Exit: use rightmost
    "cam5": True,     # Entry: use leftmost
}

# ════════════════════════════════════════════════════════════════════════════
# OCR SERVICE
# ════════════════════════════════════════════════════════════════════════════
OCR_URL = "http://127.0.0.1:9000/ocr"
OCR_TIMEOUT = 5.0

# ════════════════════════════════════════════════════════════════════════════
# PLATE VALIDATION
# ════════════════════════════════════════════════════════════════════════════
VALID_STATE_CODES = frozenset({
    "AP", "AR", "AS", "BR", "CG", "GA", "GJ", "HR", "HP", "JH", "KA", "KL",
    "MP", "MH", "MN", "ML", "MZ", "NL", "OD", "OR", "PB", "RJ", "SK", "TN",
    "TS", "TR", "UP", "UK", "UA", "WB", "AN", "CH", "DN", "DD", "DL", "JK",
    "LA", "LD", "PY"
})

STATE_CORRECTION_MAP = {'V': 'Y', 'R': 'K', 'I': 'T'}
CHAR_CORRECTION_MAP = {
    '0': 'O', 'O': '0', '1': 'I', 'I': '1',
    '5': 'S', 'S': '5', '8': 'B', 'B': '8'
}

PLATE_REGEX = re.compile(r'^[A-Z]{2}[0-9]{2}[A-Z]{1,3}[0-9]{4}$')
BH_PLATE_REGEX = re.compile(r'^[0-9]{2}BH[0-9]{4}[A-Z]{2}$')

# ════════════════════════════════════════════════════════════════════════════
# DETECTION THRESHOLDS – ✅ INCREASED FOR STABILITY
# ════════════════════════════════════════════════════════════════════════════
PLATE_CONF = 0.60
STABLE_FRAMES = 30
COOLDOWN_SEC = 30
GLOBAL_COOLDOWN_SEC = 30
LARGE_BBOX_AREA_THRESHOLD = 35000   # ⬆️ increased to avoid false JUMBO from blurry images
DB_PATH = "mongodb://localhost:27017/"

# ✅ Increased timeouts and tolerances
JOB_TIMEOUT_SEC = 120

# ✅ More forgiving stabilization
RAW_STABLE_FRAMES = 6                    # More samples for greater confidence
RAW_STABLE_TOLERANCE = 80
COUNT_STABLE_FRAMES = 7

# ✅ NEW: Minimum detection thresholds to prevent stabilisation on empty/partial frames
MIN_TOP_RAW_COUNT = 15       # Minimum cylinders for top view
MIN_SIDE_RAW_COLUMNS = 5     # Minimum columns for side view

# ✅ NEW: NMS threshold to remove duplicate detections
NMS_IOU_THRESHOLD = 0.5

OCR_ENGINE = "paddle"

# ════════════════════════════════════════════════════════════════════════════
# PROCESSING FPS
# ════════════════════════════════════════════════════════════════════════════
PROCESSING_FPS = 2
FRAME_TIME = 1.0 / PROCESSING_FPS

# ════════════════════════════════════════════════════════════════════════════
# MODEL PATHS
# ════════════════════════════════════════════════════════════════════════════
PLATE_MODEL_PATH = r"C:\Users\admin\Desktop\cyl\approacg-script-backup\models\Anpr_plate.pt"
TOP_MODEL_PATH = r"C:\Users\admin\Desktop\cyl\approacg-script-backup\models\top_cam.pt"
SIDE_MODEL_PATH = r"C:\Users\admin\Desktop\cyl\approacg-script-backup\models\side_v8m.pt"

# ════════════════════════════════════════════════════════════════════════════
# OUTPUT DIRECTORIES
# ════════════════════════════════════════════════════════════════════════════
OUTPUT_DIRS = [
    "plates", "results", "anpr_images",
    "top_images", "side_images",
    "top_outputs", "side_outputs"
]

def ensure_directories():
    for d in OUTPUT_DIRS:
        os.makedirs(d, exist_ok=True)

# ════════════════════════════════════════════════════════════════════════════
# OPENCV FFMPEG SETTINGS
# ════════════════════════════════════════════════════════════════════════════
def setup_opencv_env():
    os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = (
        "rtsp_transport;tcp|stimeout;15000000|rtbufsize;1000000"
    )

# ════════════════════════════════════════════════════════════════════════════
# LOGGING
# ════════════════════════════════════════════════════════════════════════════
DEBUG = True
LOG_DIR = r"C:\Users\admin\Desktop\cyl\approacg-script-backup\logs"

# ════════════════════════════════════════════════════════════════════════════
# 🆕 CYLINDER COUNT LIMITS FOR 14.5 KG TRUCKS
# ════════════════════════════════════════════════════════════════════════════
MAX_CYLINDER_COUNT = 350          # Maximum cylinders for any 14.5‑kg truck
ROUND_UP_THRESHOLD = 15           # Round up to 350 if total >= 350 - 15 (i.e., >= 335)