import sys, time
sys.path.insert(0, r"C:\person-reidentification")
from fastapi.testclient import TestClient
from app.main import app
from app.config import settings

client = TestClient(app)
H = {"X-API-Key": settings.API_KEY}

# 1. Check cameras
cams = client.get("/api/v1/cameras", headers=H).json()
print("=== Cameras in DB ===")
for c in cams:
    print(f"  ID={c['id']}  name={c['name']}  url={c['rtsp_url']}")

if not cams:
    print("NO CAMERAS FOUND. Register a camera first.")
    sys.exit(1)

# Pick first camera
cam_id = cams[0]["id"]
print(f"\n=== Testing with camera ID: {cam_id} ===")

# 2. Start stream
res = client.post(f"/api/v1/stream/start/{cam_id}", headers=H)
print("Start result:", res.json())

# 3. Wait and check status
time.sleep(3)
st = client.get(f"/api/v1/stream/status/{cam_id}", headers=H).json()
print("Stream status:", st)

# 4. Get frame — no auth
fr = client.get(f"/api/v1/stream/frame/{cam_id}")
print(f"Frame response: code={fr.status_code}  type={fr.headers.get('content-type')}  size={len(fr.content)} bytes")
if fr.status_code != 200:
    print("ERROR:", fr.text)
else:
    # Save frame to disk so we can inspect it
    out = r"C:\person-reidentification\data\debug_frame.jpg"
    with open(out, "wb") as f:
        f.write(fr.content)
    print(f"Frame saved to: {out}")
    print("OPEN THIS FILE IN ANY IMAGE VIEWER TO SEE THE ANNOTATED FRAME")
