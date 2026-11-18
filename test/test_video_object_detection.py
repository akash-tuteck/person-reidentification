
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.services.object_categories import (
    NON_PERSON_COCO_CLASS_IDS,
    validate_object_categories
)

from app.services.video_object_detection_service import (
    VideoObjectDetectionService
)


def test_video_object_detection():

    print("\nVIDEO OBJECT DETECTION TEST")
    print("===========================")

    validate_object_categories()

    print(
        "Supported non-human categories:",
        len(NON_PERSON_COCO_CLASS_IDS)
    )

    print(
        "Person excluded:",
        "person" not in NON_PERSON_COCO_CLASS_IDS
    )

    video_id = input("\nEnter your existing video ID: ").strip()

    try:

        result = VideoObjectDetectionService.detect_objects(
            video_id=video_id
        )

        print("\nDETECTION RESULT")
        print("================")

        print("Video ID:", result["video_id"])

        print(
            "Frames processed:",
            result["total_frames_processed"]
        )

        print(
            "Total objects:",
            result["total_objects_detected"]
        )

        print("\nDetected categories:")

        for category, count in result["category_counts"].items():

            if count > 0:
                print(f"{category}: {count}")

        print("\nFrame-wise results:")

        for frame in result["frames"]:

            if frame["total_objects"] > 0:

                print(
                    f'Frame {frame["frame_number"]} | '
                    f'Timestamp: {frame["timestamp_seconds"]}s | '
                    f'Objects: {frame["total_objects"]}'
                )

                for obj in frame["objects"]:

                    print(
                        f'  {obj["category"]} | '
                        f'Confidence: {obj["confidence"]}'
                    )

        assert all(
            obj["category"] != "person"
            for frame in result["frames"]
            for obj in frame["objects"]
        )

        print("\nAnnotated frames:")
        print(result["annotated_frames_folder"])

        print("\nObject crops:")
        print(result["objects_folder"])

        print("\nMetadata:")
        print(result["metadata_path"])

        print("\nTEST PASSED")

    except Exception as error:

        print("\nTEST FAILED")
        print("Error:", error)


if __name__ == "__main__":
    test_video_object_detection()
