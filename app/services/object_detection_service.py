
import shutil

import cv2

from pathlib import Path
from ultralytics import YOLO

from app.services.image_service import ImageProcessingService

from app.services.object_categories import (
    NON_PERSON_COCO_CLASS_IDS,
    OBJECT_DETECTION_CONFIDENCE,
    OBJECT_DETECTION_IMAGE_SIZE,
)


BASE_DIR = Path(__file__).resolve().parent.parent

UPLOAD_DIR = BASE_DIR / "uploads"
DETECTION_DIR = UPLOAD_DIR / "detections"
PERSON_DIR = UPLOAD_DIR / "persons"
OBJECT_DIR = UPLOAD_DIR / "objects"

DETECTION_DIR.mkdir(parents=True, exist_ok=True)
PERSON_DIR.mkdir(parents=True, exist_ok=True)
OBJECT_DIR.mkdir(parents=True, exist_ok=True)


class ObjectDetectionService:

    _model = None

    # All 79 non-human COCO categories.
    # Person (class ID 0) is intentionally excluded.
    CATEGORY_CLASS_IDS = NON_PERSON_COCO_CLASS_IDS

    @classmethod
    def get_model(cls):

        if cls._model is None:
            cls._model = YOLO("yolov8n.pt")

        return cls._model

    @classmethod
    def detect_objects(
        cls,
        image_id: str,
        object_categories: list[str] | None = None
    ):

        image_path = ImageProcessingService.get_processed_image_path(
            image_id
        )

        if image_path is None:
            raise ValueError("Processed image not found")

        image = cv2.imread(str(image_path))

        if image is None:
            raise ValueError("Unable to read processed image")

        # Default: detect all 79 non-human categories.
        if object_categories is None:
            categories = list(cls.CATEGORY_CLASS_IDS.keys())

        else:
            categories = list(dict.fromkeys(
                category.strip().lower()
                for category in object_categories
            ))

        if not categories:
            raise ValueError(
                "At least one object category is required"
            )

        # Validate requested categories.
        invalid_categories = [
            category
            for category in categories
            if category not in cls.CATEGORY_CLASS_IDS
        ]

        if invalid_categories:
            raise ValueError(
                f"Unsupported object categories: {invalid_categories}. "
                f"Person is excluded from object detection. "
                f"Supported categories: "
                f"{list(cls.CATEGORY_CLASS_IDS.keys())}"
            )

        model = cls.get_model()

        class_ids = [
            cls.CATEGORY_CLASS_IDS[category]
            for category in categories
        ]

        results = model.predict(
            source=str(image_path),
            classes=class_ids,
            conf=OBJECT_DETECTION_CONFIDENCE,
            imgsz=OBJECT_DETECTION_IMAGE_SIZE,
            verbose=False
        )

        result = results[0]

        annotated_image = result.plot()

        annotated_path = DETECTION_DIR / f"{image_id}.jpg"

        if not cv2.imwrite(
            str(annotated_path),
            annotated_image
        ):
            raise ValueError(
                "Unable to save detection image"
            )

        height, width = image.shape[:2]

        detected_objects = []

        category_counts = {
            category: 0
            for category in categories
        }

        # Create category-specific crop directories.
        category_directories = {}

        for category in categories:

            category_folder = (
                OBJECT_DIR / image_id / category
            )

            if category_folder.exists():
                shutil.rmtree(category_folder)

            category_folder.mkdir(
                parents=True,
                exist_ok=True
            )

            category_directories[category] = category_folder

        if result.boxes is not None:

            for box in result.boxes:

                class_id = int(box.cls[0].item())

                # Never allow person into object detection.
                if class_id == 0:
                    continue

                category = cls.get_category_name(class_id)

                if category is None:
                    continue

                if category not in category_counts:
                    continue

                coordinates = box.xyxy[0].cpu().numpy()

                x1, y1, x2, y2 = map(int, coordinates)

                x1 = max(0, min(x1, width))
                y1 = max(0, min(y1, height))
                x2 = max(0, min(x2, width))
                y2 = max(0, min(y2, height))

                if x2 <= x1 or y2 <= y1:
                    continue

                object_crop = image[y1:y2, x1:x2]

                object_number = (
                    category_counts[category] + 1
                )

                crop_filename = (
                    f"{category}_{object_number}.jpg"
                )

                crop_path = (
                    category_directories[category]
                    / crop_filename
                )

                if not cv2.imwrite(
                    str(crop_path),
                    object_crop
                ):
                    continue

                category_counts[category] = object_number

                confidence = float(box.conf[0].item())

                detected_objects.append({
                    "object_number": object_number,
                    "category": category,
                    "class_id": class_id,
                    "confidence": round(confidence, 4),
                    "bounding_box": {
                        "x1": x1,
                        "y1": y1,
                        "x2": x2,
                        "y2": y2
                    },
                    "crop_filename": crop_filename,
                    "crop_path": str(crop_path)
                })

        return {
            "image_id": image_id,
            "total_objects": len(detected_objects),
            "category_counts": category_counts,
            "annotated_image_path": str(annotated_path),
            "objects": detected_objects
        }

    @classmethod
    def get_category_name(cls, class_id: int):

        for category, category_id in cls.CATEGORY_CLASS_IDS.items():

            if category_id == class_id:
                return category

        return None


class PersonDetectionService(ObjectDetectionService):

    """
    Person detection remains separate from object detection.

    This preserves the existing person crop directory and response
    structure so the OSNet feature extraction pipeline continues working.
    """

    @classmethod
    def detect_persons(cls, image_id: str):

        image_path = ImageProcessingService.get_processed_image_path(
            image_id
        )

        if image_path is None:
            raise ValueError("Processed image not found")

        image = cv2.imread(str(image_path))

        if image is None:
            raise ValueError("Unable to read processed image")

        model = cls.get_model()

        results = model.predict(
            source=str(image_path),
            classes=[0],
            conf=0.35,
            imgsz=640,
            verbose=False
        )

        result = results[0]

        annotated_path = DETECTION_DIR / f"{image_id}.jpg"

        if not cv2.imwrite(
            str(annotated_path),
            result.plot()
        ):
            raise ValueError(
                "Unable to save detection image"
            )

        person_folder = PERSON_DIR / image_id

        if person_folder.exists():
            shutil.rmtree(person_folder)

        person_folder.mkdir(
            parents=True,
            exist_ok=True
        )

        height, width = image.shape[:2]

        persons = []

        if result.boxes is not None:

            for box in result.boxes:

                coordinates = box.xyxy[0].cpu().numpy()

                x1, y1, x2, y2 = map(int, coordinates)

                x1 = max(0, min(x1, width))
                y1 = max(0, min(y1, height))
                x2 = max(0, min(x2, width))
                y2 = max(0, min(y2, height))

                if x2 <= x1 or y2 <= y1:
                    continue

                person_crop = image[y1:y2, x1:x2]

                person_number = len(persons) + 1

                crop_filename = (
                    f"person_{person_number}.jpg"
                )

                crop_path = (
                    person_folder / crop_filename
                )

                if not cv2.imwrite(
                    str(crop_path),
                    person_crop
                ):
                    continue

                persons.append({
                    "person_number": person_number,
                    "confidence": round(
                        float(box.conf[0].item()), 4
                    ),
                    "bounding_box": {
                        "x1": x1,
                        "y1": y1,
                        "x2": x2,
                        "y2": y2
                    },
                    "crop_filename": crop_filename
                })

        return {
            "image_id": image_id,
            "total_persons": len(persons),
            "annotated_image_path": str(annotated_path),
            "persons": persons
        }
