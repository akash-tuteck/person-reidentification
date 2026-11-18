import sys
from pathlib import Path

# Add project root to Python import path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.services.video_object_tracking_service import (
    VideoObjectTrackingService
)

VIDEO_ID = "b4b6769d-120d-4f32-bfd6-9e64da402293"


def main():
    print("\nVIDEO OBJECT TRACKING TEST")
    print("==========================")

    try:
        result = VideoObjectTrackingService.track_objects(VIDEO_ID)

        print("\nTRACKING RESULT")
        print("================")
        print("Video ID:", result["video_id"])
        print("Frames processed:", result["frames_processed"])
        print(
            "Total object detections:",
            result["total_object_detections"]
        )
        print("Total object tracks:", result["total_object_tracks"])

        print("\nCATEGORY TRACK COUNTS")
        print("=====================")

        for category, count in sorted(
            result["category_track_counts"].items()
        ):
            print(f"{category}: {count}")

        print("\nSAMPLE TRACKS")
        print("=============")

        for track_id, track in list(result["tracks"].items())[:10]:
            print(
                track_id,
                "| Category:", track["category"],
                "| Detections:", track["total_detections"]
            )

        print("\nMetadata saved:")
        print(result["metadata_path"])

        assert result["frames_processed"] > 0
        assert result["total_object_tracks"] > 0
        assert all(
            track["category"] != "person"
            for track in result["tracks"].values()
        )

        print("\nTEST PASSED")

    except Exception as error:
        print("\nTEST FAILED")
        print("Error:", str(error))
        raise


if __name__ == "__main__":
    main()
