
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.services.object_categories import (
    NON_PERSON_COCO_CLASS_IDS,
    validate_object_categories,
)

from app.services.object_detection_service import (
    ObjectDetectionService,
)


def test_object_detection():

    print("\nIMAGE OBJECT DETECTION TEST")
    print("===========================")

    validate_object_categories()

    print("Supported non-human categories:",
          len(NON_PERSON_COCO_CLASS_IDS))

    print("Person excluded:",
          "person" not in NON_PERSON_COCO_CLASS_IDS)

    # Replace this with an existing processed image ID.
    image_id = input(
        "\nEnter your existing processed image ID: "
    ).strip()

    try:

        result = ObjectDetectionService.detect_objects(
            image_id=image_id
        )

        print("\nDETECTION RESULT")
        print("================")

        print("Image ID:", result["image_id"])
        print("Total objects:", result["total_objects"])

        print("\nCategory counts:")

        detected_categories = {
            category: count
            for category, count
            in result["category_counts"].items()
            if count > 0
        }

        for category, count in detected_categories.items():
            print(f"{category}: {count}")

        print("\nDetected objects:")

        for obj in result["objects"]:

            print(
                f'{obj["category"]} | '
                f'Confidence: {obj["confidence"]} | '
                f'Crop: {obj["crop_filename"]}'
            )

        print("\nAnnotated image:",
              result["annotated_image_path"])

        assert all(
            obj["category"] != "person"
            for obj in result["objects"]
        )

        print("\nTEST PASSED")

    except Exception as error:

        print("\nTEST FAILED")
        print("Error:", error)


if __name__ == "__main__":
    test_object_detection()
