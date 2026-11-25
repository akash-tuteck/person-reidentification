
from pathlib import Path

import json
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from app.services.video_service import VideoProcessingService


video_path = Path("app/uploads/videos/watermarked_preview.mp4")

if not video_path.exists():
    raise FileNotFoundError(
        "Please put watermarked_preview.mp4 in the project root"
    )

with open(video_path, "rb") as video_file:
    video_bytes = video_file.read()

result = VideoProcessingService.process_video(
    file_bytes=video_bytes,
    filename=video_path.name,
    frame_interval_seconds=1.0
)

print("\nVIDEO PROCESSING RESULTS")
print("========================")

print("Video ID:", result["video_id"])
print("FPS:", result["fps"])
print("Duration:", result["duration_seconds"], "seconds")
print("Original frames:", result["total_video_frames"])
print("Extracted frames:", result["total_extracted_frames"])
print("Status:", result["status"])

print("\nEXTRACTED FRAMES")

for frame in result["extracted_frames"]:

    print(
        frame["frame_filename"],
        "| Timestamp:",
        frame["timestamp_ms"],
        "ms"
    )

print("\nSaved frames at:")
print(result["frames_directory"])
