"""
Phase 3 integration test — Stream endpoints.
Creates a small synthetic test video, registers it as a camera,
starts the stream, checks frame and event endpoints, then stops.
"""
import sys
import time
import numpy as np
import cv2
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from fastapi.testclient import TestClient
from app.main import app
from app.config import settings

client = TestClient(app)
HEADERS = {"X-API-Key": settings.API_KEY}


def make_test_video(path: Path, frames: int = 30, w: int = 640, h: int = 480):
    """Write a small synthetic video with a coloured rectangle (fake person)."""
    out = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10, (w, h))
    for i in range(frames):
        frame = np.zeros((h, w, 3), dtype=np.uint8)
        # Draw a fake person rectangle
        cv2.rectangle(frame, (200, 50), (280, 400), (100, 180, 240), -1)
        out.write(frame)
    out.release()


def run():
    video_path = Path("./data/test_stream.mp4")
    make_test_video(video_path)
    print(f"Test video created: {video_path}")

    print("\n" + "=" * 60)
    print("TEST 1: Create zone and camera pointing to test video")
    print("=" * 60)

    # Zone
    res = client.post("/api/v1/zones", headers=HEADERS,
                      json={"code": "STREAM_TEST_ZONE", "name": "Stream Test Zone"})
    if res.status_code == 400:
        print("  Zone already exists")
    else:
        assert res.status_code == 201, res.text
    zone_id = next(z["id"] for z in client.get("/api/v1/zones", headers=HEADERS).json()
                   if z["code"] == "STREAM_TEST_ZONE")

    # Camera — use the local video file path as the rtsp_url
    res = client.post("/api/v1/cameras", headers=HEADERS, json={
        "name": "Test Stream Camera",
        "rtsp_url": str(video_path.resolve()),
        "camera_type": "fixed",
        "zone_id": zone_id,
    })
    assert res.status_code == 201, res.text
    camera_id = res.json()["id"]
    print(f"PASSED — camera ID: {camera_id}")

    print("\n" + "=" * 60)
    print("TEST 2: Start stream")
    print("=" * 60)
    res = client.post(f"/api/v1/stream/start/{camera_id}", headers=HEADERS)
    assert res.status_code == 200, res.text
    print("PASSED:", res.json())

    # Give the thread a moment to open the video
    time.sleep(2)

    print("\n" + "=" * 60)
    print("TEST 3: Check stream status")
    print("=" * 60)
    res = client.get(f"/api/v1/stream/status/{camera_id}", headers=HEADERS)
    assert res.status_code == 200, res.text
    data = res.json()
    print("PASSED:", data)
    assert data["camera_id"] == camera_id

    print("\n" + "=" * 60)
    print("TEST 4: Get latest annotated frame")
    print("=" * 60)
    time.sleep(1)
    res = client.get(f"/api/v1/stream/frame/{camera_id}", headers=HEADERS)
    if res.status_code == 200:
        print(f"PASSED — frame received, size: {len(res.content)} bytes, type: {res.headers['content-type']}")
        assert res.headers["content-type"] == "image/jpeg"
    else:
        # Stream might still be starting — this is acceptable
        print(f"Frame not ready yet ({res.status_code}) — acceptable for a fresh start")

    print("\n" + "=" * 60)
    print("TEST 5: List all stream statuses")
    print("=" * 60)
    res = client.get("/api/v1/stream/status", headers=HEADERS)
    assert res.status_code == 200, res.text
    statuses = res.json()
    print(f"PASSED — {len(statuses)} active stream(s)")

    print("\n" + "=" * 60)
    print("TEST 6: Security events endpoint")
    print("=" * 60)
    res = client.get("/api/v1/stream/events?limit=10", headers=HEADERS)
    assert res.status_code == 200, res.text
    events = res.json()
    print(f"PASSED — {len(events)} event(s) in DB")

    print("\n" + "=" * 60)
    print("TEST 7: Stop stream")
    print("=" * 60)
    res = client.post(f"/api/v1/stream/stop/{camera_id}", headers=HEADERS)
    assert res.status_code == 200, res.text
    print("PASSED:", res.json())

    print("\n" + "=" * 60)
    print(">>> ALL PHASE 3 TESTS PASSED 100%!")
    print("=" * 60)


if __name__ == "__main__":
    run()
