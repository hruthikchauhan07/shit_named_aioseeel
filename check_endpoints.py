#!/usr/bin/env python
"""
Test OCR Server - Send an image to both OCR workers and print responses.
Usage: python test_ocr.py <image_path>
"""

import sys
import requests
import json
import time
from pathlib import Path

# OCR server endpoints
OCR_ENDPOINTS = [
    "http://127.0.0.1:8090/ocr",
    "http://127.0.0.1:8091/ocr",
]
HEALTH_ENDPOINTS = [
    "http://127.0.0.1:8090/health",
    "http://127.0.0.1:8091/health",
]

def check_health():
    """Check health status of both OCR servers"""
    print("\n=== OCR Server Health Check ===")
    for url in HEALTH_ENDPOINTS:
        try:
            resp = requests.get(url, timeout=5)
            data = resp.json()
            print(f"{url} → status={resp.status_code}, ready={data.get('ready')}, pid={data.get('pid')}")
        except Exception as e:
            print(f"{url} → ERROR: {e}")

def test_ocr(image_path: str):
    """Send image to both OCR endpoints and print results"""
    if not Path(image_path).exists():
        print(f"Error: Image file not found: {image_path}")
        sys.exit(1)

    print(f"\n=== Testing OCR with image: {image_path} ===")
    with open(image_path, "rb") as f:
        files = {"file": (image_path, f, "image/jpeg")}
        
        for idx, url in enumerate(OCR_ENDPOINTS):
            print(f"\n--- Request to {url} ---")
            start = time.time()
            try:
                response = requests.post(url, files=files, timeout=30)
                elapsed = time.time() - start
                print(f"HTTP Status: {response.status_code}")
                print(f"Response time: {elapsed:.3f}s")
                
                if response.status_code == 200:
                    result = response.json()
                    print(result)
                    plate = result.get("plate")
                    print(f"OCR Result: {plate}")
                    if plate is None:
                        print("Warning: OCR returned None (no plate recognized)")
                else:
                    print(f"Error: {response.text}")
            except requests.exceptions.Timeout:
                print("Error: Request timed out after 30 seconds")
            except Exception as e:
                print(f"Error: {e}")
            
            # Small delay between requests to avoid overwhelming
            time.sleep(0.5)

def main():
    if len(sys.argv) < 2:
        print("Usage: python test_ocr.py <image_path>")
        print("Example: python test_ocr.py plates/MH12AB1234_20250428.jpg")
        sys.exit(1)
    
    image_path = sys.argv[1]
    
    # First check health
    check_health()
    
    # Then test OCR
    test_ocr(image_path)
    
    print("\n=== Done ===")

if __name__ == "__main__":
    main()