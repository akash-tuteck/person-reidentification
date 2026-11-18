
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

sys.path.insert(0, str(PROJECT_ROOT))

from app.services.video_object_feature_service import (
    VideoObjectFeatureService
)


def test_video_object_features():

    print("\nVIDEO OBJECT FEATURE EXTRACTION")
    print("================================")

    print(
        "Supported categories:",
        len(VideoObjectFeatureService.SUPPORTED_CATEGORIES)
    )

    print(
        "Embedding dimension:",
        VideoObjectFeatureService.EMBEDDING_DIMENSION
    )

    video_id = input("\nEnter your video ID: ").strip()

    try:

        result = VideoObjectFeatureService.extract_features(
            video_id
        )

        print("\nFEATURE EXTRACTION RESULT")
        print("=========================")

        print("Video ID:", result["video_id"])

        print(
            "Frames processed:",
            result["total_frames_processed"]
        )

        print(
            "Objects processed:",
            result["total_objects_processed"]
        )

        print(
            "Embedding dimension:",
            result["embedding_dimension"]
        )

        print("\nCategory counts:")

        for category, count in result["category_counts"].items():
            print(f"{category}: {count}")

        print("\nEmbedding folder:")
        print(result["embedding_folder"])

        print("\nMetadata:")
        print(result["metadata_path"])

        assert result["embedding_dimension"] == 768

        assert all(
            obj["category"] != "person"
            for obj in result["processed_objects"]
        )

        assert all(
            obj["embedding_dimension"] == 768
            for obj in result["processed_objects"]
        )

        print("\nTEST PASSED")

    except Exception as error:

        print("\nTEST FAILED")
        print("Error:", error)


if __name__ == "__main__":
    test_video_object_features()
