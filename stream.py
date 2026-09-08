"""
capture_rtsp_snapshots.py

Connects to each RTSP camera one by one, captures a single frame,
and saves it to an output folder.

Requirements:
    pip install opencv-python

If OpenCV has issues opening RTSP streams, install FFmpeg/GStreamer support
or use an OpenCV build with FFmpeg enabled.
"""

import os
import cv2
from datetime import datetime

# ============================================================
# Camera Configuration
# ============================================================

BASE_IP = "192.168.3.250"
USERNAME = "admin"
PASSWORD = "Iocl%401234"


def rtsp_url(channel, subtype=0):
    return (
        f"rtsp://{USERNAME}:{PASSWORD}@{BASE_IP}:554"
        f"/cam/realmonitor?channel={channel}&subtype={subtype}"
    )


CAMERA_CONFIG = {
    "cam1": rtsp_url(4, 0),  # top view 1
    "cam2": rtsp_url(2, 0),  # exit ANPR
    "cam3": rtsp_url(3, 0),  # entry ANPR
    "cam4": rtsp_url(1, 0),  # side view 1
    "cam5": rtsp_url(5, 0),  # top view 2
    "cam6": rtsp_url(6, 0),  # side view 2
}

OUTPUT_DIR = "camera_snapshots"

# Wait this many frames before saving (helps RTSP stabilize)
WARMUP_FRAMES = 15

# ============================================================


def capture_frame(rtsp_link):
    cap = cv2.VideoCapture(rtsp_link, cv2.CAP_FFMPEG)

    if not cap.isOpened():
        print("  Could not connect.")
        return None

    frame = None

    # Read a few frames to stabilize the stream
    for _ in range(WARMUP_FRAMES):
        ret, frame = cap.read()
        if not ret:
            frame = None
            break

    cap.release()

    return frame


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    print("=" * 60)
    print("Capturing images from RTSP cameras...")
    print("=" * 60)

    for cam_name, rtsp in CAMERA_CONFIG.items():

        print(f"\n[{cam_name}]")
        print(rtsp)

        frame = capture_frame(rtsp)

        if frame is None:
            print("  Failed to capture frame.")
            continue

        filename = f"{cam_name}_{timestamp}.jpg"
        save_path = os.path.join(OUTPUT_DIR, filename)

        cv2.imwrite(save_path, frame)

        h, w = frame.shape[:2]
        print(f"  Saved: {save_path}")
        print(f"  Resolution: {w} x {h}")

    print("\nDone.")


if __name__ == "__main__":
    main()