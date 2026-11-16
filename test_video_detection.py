
from app.services.video_detection_service import VideoDetectionService


VIDEO_ID = "b4b6769d-120d-4f32-bfd6-9e64da402293"


print("\nVIDEO OBJECT DETECTION")
print("======================")

result = VideoDetectionService.detect_video_frames(
    video_id=VIDEO_ID
)

print("Video ID:", result["video_id"])
print("Frames processed:", result["total_frames_processed"])
print("Total objects detected:", result["total_objects_detected"])
print("Total persons detected:", result["total_persons_detected"])
print("Categories:", result["categories_processed"])
print("Status:", result["status"])

print("\nFRAME-WISE RESULTS")
print("==================")

for frame in result["results"]:

    print(
        f"\nFrame: {frame['frame_number']} "
        f"| Timestamp: {frame['timestamp_ms']} ms"
    )

    print("Objects:", frame["total_objects"])
    print("Persons:", frame["total_persons"])
    print("Categories:", frame["category_counts"])
