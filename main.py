import sys
import signal
import threading
import time
import logging

# Initialisation must happen before importing globals
from config import ensure_directories, setup_opencv_env, DEBUG
from config import STABLE_FRAMES, JOB_TIMEOUT_SEC, CAMERA_CONFIG, OCR_URL, PLATE_MODEL_PATH, TOP_MODEL_PATH, SIDE_MODEL_PATH
from logger_setup import setup_logging

setup_logging(DEBUG)
setup_opencv_env()
ensure_directories()

# Now import the shared state and processor modules
import state
from plate_monitor import plate_monitor
from top_processor import top_processor
from side_processor import side_processor
from watchdog import job_watchdog

logger = logging.getLogger(__name__)


def signal_handler(signum, frame):
    """Handle shutdown signals gracefully"""
    logger.info(f"Received signal {signum}, shutting down...")
    state.STOP_EVENT.set()


def main():
    logger.info("=" * 60)
    logger.info("SYSTEM STARTING")
    logger.info("=" * 60)

    # Log configuration for debugging
    logger.info(f"Configuration: DEBUG={DEBUG}, STABLE_FRAMES={STABLE_FRAMES}, JOB_TIMEOUT_SEC={JOB_TIMEOUT_SEC}")
    logger.info(f"Camera config: {list(CAMERA_CONFIG.keys())}")
    logger.info(f"OCR URL: {OCR_URL}")
    logger.info(f"Model paths: PLATE={PLATE_MODEL_PATH}, TOP={TOP_MODEL_PATH}, SIDE={SIDE_MODEL_PATH}")

    # Verify directories exist
    import os
    for dir_name in ["anpr_images", "plates", "top_images", "top_outputs", "side_images", "side_outputs", "logs"]:
        if os.path.exists(dir_name):
            logger.debug(f"Directory exists: {dir_name}")
        else:
            logger.warning(f"Directory missing: {dir_name}")

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    threads = [
        threading.Thread(target=plate_monitor, args=("cam3",), daemon=True, name="anpr-cam3"),
        threading.Thread(target=plate_monitor, args=("cam5",), daemon=True, name="anpr-cam5"),
        threading.Thread(target=top_processor, daemon=True, name="top-proc"),
        threading.Thread(target=side_processor, daemon=True, name="side-proc"),
        threading.Thread(target=job_watchdog, daemon=True, name="watchdog"),
    ]

    for t in threads:
        t.start()
        logger.info(f"Started thread: {t.name}")

    logger.info("System started. Press Ctrl+C to stop.")
    logger.info("cam3=EXIT (FRONT view), cam5=ENTRY (REAR view)")

    try:
        while not state.STOP_EVENT.is_set():
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received")
    finally:
        logger.info("Shutting down...")
        state.STOP_EVENT.set()
        for t in threads:
            t.join(timeout=5)
        logger.info("System stopped cleanly")

    return 0


if __name__ == "__main__":
    sys.exit(main())