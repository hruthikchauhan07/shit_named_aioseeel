#!/usr/bin/env python3
"""
RTSP Test Script for INFINOVA Cameras (NVR C++ Platform)

Uses the correct path format: /media/videoX
"""

import cv2
import time
import os
import urllib.parse

# ─── Configuration ──────────────────────────────────────────────────────────
USERNAME = "admin"
PASSWORD_RAW = "Iocl@1234"
PASSWORD_ENCODED = urllib.parse.quote(PASSWORD_RAW, safe="")
PORT = 8554
TIMEOUT_SEC = 5.0

# List of IPs to test
CAMERA_IPS = [
    # "192.168.3.116",
    # "192.168.3.122",
    # "192.168.3.117",
    "192.168.3.127",
]

# Channels to try (1 to 8)
MAX_CHANNELS = 8

# Output directory for test frames
OUTPUT_DIR = "rtsp_test_frames"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def test_infinova_url(ip: str, channel: int, main_stream: bool) -> bool:
    """
    Test an Infinova RTSP URL.

    Main stream:   video index = (ch - 1) * 2 + 1
    Sub stream:    video index = (ch - 1) * 2 + 2
    """
    video_index = (channel - 1) * 2 + (1 if main_stream else 2)
    path = f"/media/video{video_index}"

    # Try both encoded and raw password (FFmpeg sometimes handles raw better)
    for use_encoded in [True, False]:
        password = PASSWORD_ENCODED if use_encoded else PASSWORD_RAW
        url = f"rtsp://{USERNAME}:{password}@{ip}:{PORT}{path}"

        # Mask password in display
        display_url = url.replace(PASSWORD_ENCODED, "***").replace(PASSWORD_RAW, "***")
        stream_type = "Main" if main_stream else "Sub"
        print(f"  Trying ch{channel} {stream_type}: {display_url[:80]}...", end=" ", flush=True)

        cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
        if not cap.isOpened():
            print("❌ Failed to open")
            continue

        # Try to read a frame
        success = False
        start = time.time()
        while time.time() - start < TIMEOUT_SEC:
            ret, frame = cap.read()
            if ret and frame is not None:
                success = True
                break
            time.sleep(0.1)

        cap.release()

        if success:
            print("✅ SUCCESS!")
            # Save a test frame
            filename = f"frame_{ip}_ch{channel}_{'main' if main_stream else 'sub'}.jpg"
            filepath = os.path.join(OUTPUT_DIR, filename)
            cv2.imwrite(filepath, frame)
            print(f"       📸 Frame saved: {filepath}")
            return True
        else:
            print("❌ Timeout / No data")

    return False


def main():
    print("=" * 70)
    print("INFINOVA RTSP STREAM TEST (/media/videoX format)")
    print(f"Username: {USERNAME}")
    print(f"Password: {PASSWORD_RAW} (encoded: {PASSWORD_ENCODED})")
    print(f"Port: {PORT}")
    print(f"Channels to test: 1 to {MAX_CHANNELS}")
    print("=" * 70)

    working_streams = []

    for ip in CAMERA_IPS:
        print(f"\n📷 Testing IP: {ip}")
        for channel in range(1, MAX_CHANNELS + 1):
            # Test main stream first
            if test_infinova_url(ip, channel, main_stream=True):
                working_streams.append((ip, channel, "main"))
            # Test sub stream
            if test_infinova_url(ip, channel, main_stream=False):
                working_streams.append((ip, channel, "sub"))

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    if working_streams:
        print(f"✅ Found {len(working_streams)} working stream(s):")
        for ip, ch, stream_type in working_streams:
            video_idx = (ch - 1) * 2 + (1 if stream_type == "main" else 2)
            print(f"  - {ip} | Channel {ch} ({stream_type}) -> /media/video{video_idx}")
            print(f"    rtsp://{USERNAME}:{PASSWORD_RAW}@{ip}:{PORT}/media/video{video_idx}")
    else:
        print("❌ No working streams found.")
        print("Tips:")
        print("  - Check if the camera is reachable (ping).")
        print("  - Verify credentials in the camera's web interface.")
        print("  - Try port 8554 if 554 doesn't work.")

    print("=" * 70)

if __name__ == "__main__":
    main()