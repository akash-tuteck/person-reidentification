
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.services.video_annotation_service import (
    VideoAnnotationService
)


VIDEO_ID = "b4b6769d-120d-4f32-bfd6-9e64da402293"


def main():
    print("\nVIDEO ANNOTATION TEST")
    print("=====================")

    result = VideoAnnotationService.generate_annotated_video(
        video_id=VIDEO_ID,
        output_fps=1.0
    )

    print("\nANNOTATION RESULT")
    print("=================")

    for key, value in result.items():
        print(f"{key}: {value}")

    output_path = Path(result["output_path"])

    assert output_path.exists()
    assert output_path.stat().st_size > 0
    assert result["frames_annotated"] > 0

    print("\nTEST PASSED")


if __name__ == "__main__":
    main()
