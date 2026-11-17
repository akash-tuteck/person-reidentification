
import sys
from pathlib import Path

# Add project root to Python import path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.services.object_detection_service import ObjectDetectionService



VIDEO_ID = "b4b6769d-120d-4f32-bfd6-9e64da402293"

FRAME_PATH = (
    Path("app/uploads/frames")
    / VIDEO_ID
    / "frame_000001.jpg"
)


def main():

    print("\nYOLO OBJECT DETECTION DIAGNOSTIC")
    print("=" * 40)

    if not FRAME_PATH.exists():
        print("Frame not found:", FRAME_PATH)
        return

    model = ObjectDetectionService.get_model()

    # Test 1: Detect every class without filtering
    results = model.predict(
        source=str(FRAME_PATH),
        conf=0.10,
        imgsz=640,
        verbose=False
    )

    result = results[0]

    print("\nTEST 1: All detected classes")
    print("-" * 40)

    if result.boxes is None or len(result.boxes) == 0:
        print("No objects detected at confidence 0.10")

    else:
        for box in result.boxes:

            class_id = int(box.cls[0].item())
            confidence = float(box.conf[0].item())

            category = result.names[class_id]

            print(
                f"Category: {category}, "
                f"Class ID: {class_id}, "
                f"Confidence: {confidence:.3f}"
            )

    # Test 2: Detect only our required non-human classes
    category_ids = [
        2, 3, 5, 7, 14, 15, 16
    ]

    filtered_results = model.predict(
        source=str(FRAME_PATH),
        classes=category_ids,
        conf=0.10,
        imgsz=640,
        verbose=False
    )

    filtered_result = filtered_results[0]

    print("\nTEST 2: Required non-human objects")
    print("-" * 40)

    if (
        filtered_result.boxes is None
        or len(filtered_result.boxes) == 0
    ):
        print("No required non-human objects detected")

    else:
        for box in filtered_result.boxes:

            class_id = int(box.cls[0].item())
            confidence = float(box.conf[0].item())

            print(
                f"Category: {filtered_result.names[class_id]}, "
                f"Confidence: {confidence:.3f}"
            )

    print("\nDiagnostic completed.")


if __name__ == "__main__":
    main()
