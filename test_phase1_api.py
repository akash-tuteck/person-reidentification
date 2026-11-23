"""
Phase 1 + Phase 2 integration test.
Run from project root: venv\Scripts\python.exe test_phase1_api.py
"""
import io
import os
import sys
from pathlib import Path
import numpy as np
import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent))

from fastapi.testclient import TestClient
from app.main import app
from app.config import settings

client = TestClient(app)
HEADERS = {"X-API-Key": settings.API_KEY}


def make_person_image(h=256, w=128):
    """Create a checkerboard image large enough to pass quality check."""
    img = np.zeros((h, w, 3), dtype=np.uint8)
    for y in range(0, h, 16):
        for x in range(0, w, 16):
            color = [200, 150, 100] if (x // 16 + y // 16) % 2 == 0 else [50, 100, 150]
            img[y:y+16, x:x+16] = color
    _, buf = cv2.imencode(".jpg", img)
    return buf.tobytes()


def run():
    print("\n" + "=" * 60)
    print("TEST 1: Health check")
    print("=" * 60)
    res = client.get("/health")
    assert res.status_code == 200
    print("PASSED:", res.json())

    print("\n" + "=" * 60)
    print("TEST 2: API key blocks unauthorized access")
    print("=" * 60)
    assert client.get("/api/v1/zones").status_code == 401
    assert client.get("/api/v1/zones", headers={"X-API-Key": "wrong"}).status_code == 401
    print("PASSED — 401 returned correctly")

    print("\n" + "=" * 60)
    print("TEST 3: Create zones")
    print("=" * 60)
    for code, name in [("T_SERVER_ROOM", "Server Room"), ("T_CONTROL_ROOM", "Control Room")]:
        res = client.post("/api/v1/zones", headers=HEADERS,
                          json={"code": code, "name": name})
        if res.status_code == 400 and "already exists" in res.text:
            print(f"  Zone {code} already exists — skipping")
        else:
            assert res.status_code == 201, res.text
            print(f"  Created zone: {res.json()}")

    res = client.get("/api/v1/zones", headers=HEADERS)
    assert res.status_code == 200
    zones = res.json()
    zone_map = {z["code"]: z["id"] for z in zones}
    assert "T_SERVER_ROOM" in zone_map
    print(f"PASSED — {len(zones)} zones in DB")

    print("\n" + "=" * 60)
    print("TEST 4: Register camera with RTSP masking")
    print("=" * 60)
    res = client.post("/api/v1/cameras", headers=HEADERS, json={
        "name": "Test Camera",
        "rtsp_url": "rtsp://admin:secret@192.168.1.5:554/stream1",
        "camera_type": "fixed",
        "zone_id": zone_map["T_SERVER_ROOM"],
    })
    assert res.status_code == 201, res.text
    cam = res.json()
    assert "secret" not in cam["rtsp_url"]
    assert "***" in cam["rtsp_url"]
    print(f"PASSED — RTSP masked: {cam['rtsp_url']}")

    print("\n" + "=" * 60)
    print("TEST 5: Enroll person in T_SERVER_ROOM (ALLOWED zone)")
    print("=" * 60)
    img_bytes = make_person_image()
    photo = ("photo.jpg", io.BytesIO(img_bytes), "image/jpeg")
    emp = f"TEST-{np.random.randint(10000, 99999)}"
    res = client.post("/api/v1/persons/enroll", headers=HEADERS,
        data={"name": "Test Person", "employee_code": emp,
              "department": "IT", "allowed_zone_codes": "T_SERVER_ROOM"},
        files=[("photo_1", photo)])
    assert res.status_code == 201, res.text
    person = res.json()
    person_id = person["id"]
    assert person["photos_count"] == 1
    assert "T_SERVER_ROOM" in person["allowed_zones"]
    print(f"PASSED — enrolled: {person}")

    print("\n" + "=" * 60)
    print("TEST 6: List persons")
    print("=" * 60)
    res = client.get("/api/v1/persons", headers=HEADERS)
    assert res.status_code == 200
    ids = [p["id"] for p in res.json()]
    assert person_id in ids
    print(f"PASSED — {len(ids)} person(s) in DB")

    # =========================================================
    # PHASE 2 TESTS
    # =========================================================

    print("\n" + "=" * 60)
    print("TEST 7: IDENTIFY — same photo → should return ALLOWED")
    print("=" * 60)
    photo2 = ("photo.jpg", io.BytesIO(img_bytes), "image/jpeg")
    res = client.post("/api/v1/identify", headers=HEADERS,
        data={"zone_code": "T_SERVER_ROOM"},
        files=[("photo", photo2)])
    assert res.status_code == 200, res.text
    result = res.json()
    print(f"  Status: {result['status']} | Color: {result['color']}")
    print(f"  Message: {result['message']}")
    print(f"  Score: {result['similarity_score']}")
    assert result["status"] in ("ALLOWED", "VERIFY"), f"Expected ALLOWED/VERIFY, got {result['status']}"
    print("PASSED")

    print("\n" + "=" * 60)
    print("TEST 8: IDENTIFY — same person, WRONG zone → should return UNAUTHORIZED")
    print("=" * 60)
    photo3 = ("photo.jpg", io.BytesIO(img_bytes), "image/jpeg")
    res = client.post("/api/v1/identify", headers=HEADERS,
        data={"zone_code": "T_CONTROL_ROOM"},
        files=[("photo", photo3)])
    assert res.status_code == 200, res.text
    result = res.json()
    print(f"  Status: {result['status']} | Color: {result['color']}")
    print(f"  Message: {result['message']}")
    assert result["status"] in ("UNAUTHORIZED", "VERIFY", "UNKNOWN"), \
        f"Unexpected status: {result['status']}"
    print("PASSED")

    print("\n" + "=" * 60)
    print("TEST 9: IDENTIFY — tiny image → should return NO_DECISION")
    print("=" * 60)
    tiny_img = np.zeros((30, 20, 3), dtype=np.uint8)
    _, tiny_buf = cv2.imencode(".jpg", tiny_img)
    tiny_photo = ("tiny.jpg", io.BytesIO(tiny_buf.tobytes()), "image/jpeg")
    res = client.post("/api/v1/identify", headers=HEADERS,
        data={"zone_code": "T_SERVER_ROOM"},
        files=[("photo", tiny_photo)])
    assert res.status_code == 200, res.text
    result = res.json()
    print(f"  Status: {result['status']} | Message: {result['message']}")
    assert result["status"] == "NO_DECISION"
    print("PASSED")

    print("\n" + "=" * 60)
    print("TEST 10: IDENTIFY — invalid zone → should return ERROR")
    print("=" * 60)
    photo4 = ("photo.jpg", io.BytesIO(img_bytes), "image/jpeg")
    res = client.post("/api/v1/identify", headers=HEADERS,
        data={"zone_code": "ZONE_THAT_DOES_NOT_EXIST"},
        files=[("photo", photo4)])
    assert res.status_code == 200, res.text
    result = res.json()
    print(f"  Status: {result['status']} | Message: {result['message']}")
    assert result["status"] == "ERROR"
    print("PASSED")

    print("\n" + "=" * 60)
    print("TEST 11: Delete person — cleanup")
    print("=" * 60)
    res = client.delete(f"/api/v1/persons/{person_id}", headers=HEADERS)
    assert res.status_code == 200
    remaining = [p["id"] for p in client.get("/api/v1/persons", headers=HEADERS).json()]
    assert person_id not in remaining
    print(f"PASSED — person {person_id} deleted")

    print("\n" + "=" * 60)
    print("TEST 12: Baseline accuracy check (existing SimilarityService)")
    print("=" * 60)
    import subprocess
    result = subprocess.run(
        [str(Path(__file__).parent / "venv" / "Scripts" / "python.exe"),
         str(Path(__file__).parent / "test" / "test_matching_scenarios.py")],
        env={**os.environ, "PYTHONPATH": str(Path(__file__).parent)},
        capture_output=True, text=True
    )
    print(result.stdout)
    assert result.returncode == 0, f"Baseline FAILED:\n{result.stderr}"
    print("PASSED — existing accuracy unchanged")

    print("\n" + "=" * 60)
    print(">>> ALL PHASE 1 + PHASE 2 TESTS PASSED 100%! <<<")
    print("=" * 60)


if __name__ == "__main__":
    run()
