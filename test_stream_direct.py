"""
Direct stream test — runs the stream in-process and saves frames to disk.
Run this SEPARATELY from the uvicorn server.
"""
import sys, time, cv2
sys.path.insert(0, r"C:\person-reidentification")

from app.database.session import SessionLocal, init_db
from app.services.stream_service import start_stream, get_latest_frame, stop_stream

init_db()

CAMERA_ID = 4  # Your CCTV camera

print(f"Starting stream for camera {CAMERA_ID}...")
result = start_stream(CAMERA_ID)
print("Result:", result)

print("Waiting for first frame (up to 15 seconds)...")
frame = None
for i in range(30):
    time.sleep(0.5)
    frame = get_latest_frame(CAMERA_ID)
    if frame is not None:
        print(f"Got frame after {(i+1)*0.5:.1f}s — shape: {frame.shape}")
        break
    else:
        print(f"  {(i+1)*0.5:.1f}s — no frame yet...")

if frame is not None:
    out_path = r"C:\person-reidentification\data\live_test_frame.jpg"
    cv2.imwrite(out_path, frame)
    print(f"\nFrame saved to: {out_path}")
    print("Open this file to see the annotated frame!")
else:
    print("ERROR: No frame received after 15 seconds")

stop_stream(CAMERA_ID)
print("Stream stopped.")
