
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.services.object_categories import (
    COCO_CLASS_NAMES,
    NON_PERSON_COCO_CLASS_IDS,
    NON_PERSON_CATEGORIES,
    validate_object_categories,
)


def test_object_categories():

    print("\nOBJECT CATEGORY CONFIGURATION")
    print("=============================")

    print("Total COCO categories:", len(COCO_CLASS_NAMES))
    print("Non-human categories:", len(NON_PERSON_CATEGORIES))
    print("Person excluded:", "person" not in NON_PERSON_COCO_CLASS_IDS)

    validate_object_categories()

    print("\nAvailable non-human categories:")

    for category, class_id in NON_PERSON_COCO_CLASS_IDS.items():
        print(f"{class_id}: {category}")

    print("\nConfiguration validation successful.")


if __name__ == "__main__":
    test_object_categories()
