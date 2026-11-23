
import cv2

from pathlib import Path
from ultralytics import YOLO

from app.config import settings
from app.services.image_service import ImageProcessingService


BASE_DIR = Path(__file__).resolve().parent.parent

UPLOAD_DIR = BASE_DIR / "uploads"
DETECTION_DIR = UPLOAD_DIR / "detections"
PERSON_DIR = UPLOAD_DIR / "persons"

DETECTION_DIR.mkdir(parents=True, exist_ok=True)
PERSON_DIR.mkdir(parents=True, exist_ok=True)


class PersonDetectionService:

    _model = None

    @classmethod
    def get_model(cls):
        if cls._model is None:
            cls._model = YOLO(settings.YOLO_MODEL)

        return cls._model

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
            conf=settings.YOLO_CONF,
            imgsz=settings.YOLO_IMGSZ,
            verbose=False
        )

        result = results[0]

        annotated_image = result.plot()

        annotated_path = DETECTION_DIR / f"{image_id}.jpg"

        if not cv2.imwrite(str(annotated_path), annotated_image):
            raise ValueError("Unable to save detection image")

        person_folder = PERSON_DIR / image_id
        person_folder.mkdir(parents=True, exist_ok=True)

        height, width = image.shape[:2]

        persons = []

        if result.boxes is not None:
            for index, box in enumerate(result.boxes):

                coordinates = box.xyxy[0].cpu().numpy()

                x1, y1, x2, y2 = map(
                    int,
                    coordinates
                )

                x1 = max(0, min(x1, width))
                y1 = max(0, min(y1, height))
                x2 = max(0, min(x2, width))
                y2 = max(0, min(y2, height))

                if x2 <= x1 or y2 <= y1:
                    continue

                person_crop = image[y1:y2, x1:x2]

                crop_filename = f"person_{index + 1}.jpg"
                crop_path = person_folder / crop_filename

                if not cv2.imwrite(str(crop_path), person_crop):
                    continue

                confidence = float(box.conf[0].item())

                persons.append({
                    "person_number": len(persons) + 1,
                    "confidence": round(confidence, 4),
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
