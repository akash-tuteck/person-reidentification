
import json
import shutil
import cv2

from pathlib import Path

from app.services.object_detection_service import (
    ObjectDetectionService
)

from app.services.object_categories import (
    NON_PERSON_COCO_CLASS_IDS,
    OBJECT_DETECTION_CONFIDENCE,
    OBJECT_DETECTION_IMAGE_SIZE
)


BASE_DIR = Path(__file__).resolve().parent.parent

UPLOAD_DIR = BASE_DIR / "uploads"

FRAME_DIR = UPLOAD_DIR / "frames"

VIDEO_OBJECT_DIR = UPLOAD_DIR / "video_objects"

VIDEO_OBJECT_DETECTION_DIR = (
    UPLOAD_DIR / "video_object_detections"
)


class VideoObjectDetectionService:

    # Reuse the existing YOLO model.
    _model = None

    # All 79 non-human COCO categories.
    CATEGORY_CLASS_IDS = NON_PERSON_COCO_CLASS_IDS

    CONFIDENCE_THRESHOLD = OBJECT_DETECTION_CONFIDENCE

    IMAGE_SIZE = OBJECT_DETECTION_IMAGE_SIZE

    @classmethod
    def get_model(cls):

        if cls._model is None:
            cls._model = ObjectDetectionService.get_model()

        return cls._model

    @staticmethod
    def read_json(path):

        with path.open("r", encoding="utf-8") as file:
            return json.load(file)

    @classmethod
    def detect_objects(
        cls,
        video_id: str,
        object_categories: list[str] | None = None
    ):

        video_id = str(video_id)

        frame_metadata_path = (
            FRAME_DIR / video_id / "frames_metadata.json"
        )

        if not frame_metadata_path.exists():
            raise FileNotFoundError(
                "Extract video frames before detecting objects"
            )

        frame_metadata = cls.read_json(
            frame_metadata_path
        )

        frames = frame_metadata.get("frames", [])

        if not frames:
            raise ValueError(
                "No extracted frames found for this video"
            )

        # Default: all 79 non-human categories.
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

        invalid_categories = [
            category
            for category in categories
            if category not in cls.CATEGORY_CLASS_IDS
        ]

        if invalid_categories:
            raise ValueError(
                f"Unsupported categories: {invalid_categories}. "
                f"Supported categories: "
                f"{list(cls.CATEGORY_CLASS_IDS.keys())}"
            )

        # Remove previous output for this video only.
        video_object_dir = VIDEO_OBJECT_DIR / video_id

        video_detection_dir = (
            VIDEO_OBJECT_DETECTION_DIR / video_id
        )

        if video_object_dir.exists():
            shutil.rmtree(video_object_dir)

        if video_detection_dir.exists():
            shutil.rmtree(video_detection_dir)

        video_object_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        annotated_dir = (
            video_detection_dir / "annotated_frames"
        )

        annotated_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        model = cls.get_model()

        class_ids = [
            cls.CATEGORY_CLASS_IDS[category]
            for category in categories
        ]

        frame_results = []

        total_objects = 0

        total_category_counts = {
            category: 0
            for category in categories
        }

        for frame_info in frames:

            frame_filename = frame_info.get(
                "frame_filename"
            )

            if not frame_filename:
                raise ValueError(
                    "Frame metadata is missing frame_filename"
                )

            frame_number = int(
                frame_info.get("frame_number", 0)
            )

            timestamp = float(
                frame_info.get("timestamp_seconds", 0)
            )

            frame_path = (
                FRAME_DIR / video_id / frame_filename
            )

            if not frame_path.exists():
                raise FileNotFoundError(
                    f"Video frame not found: {frame_path}"
                )

            image = cv2.imread(str(frame_path))

            if image is None:
                raise ValueError(
                    f"Unable to read video frame: {frame_path}"
                )

            height, width = image.shape[:2]

            results = model.predict(
                source=str(frame_path),
                classes=class_ids,
                conf=cls.CONFIDENCE_THRESHOLD,
                imgsz=cls.IMAGE_SIZE,
                verbose=False
            )

            result = results[0]

            frame_stem = Path(frame_filename).stem

            frame_object_dir = (
                video_object_dir / frame_stem
            )

            frame_object_dir.mkdir(
                parents=True,
                exist_ok=True
            )

            category_directories = {}

            for category in categories:

                category_dir = (
                    frame_object_dir / category
                )

                category_dir.mkdir(
                    parents=True,
                    exist_ok=True
                )

                category_directories[category] = category_dir

            category_counts = {
                category: 0
                for category in categories
            }

            detected_objects = []

            if result.boxes is not None:

                for box in result.boxes:

                    class_id = int(
                        box.cls[0].item()
                    )

                    # Explicitly exclude people.
                    if class_id == 0:
                        continue

                    category = result.names[class_id]

                    if category not in category_counts:
                        continue

                    coordinates = (
                        box.xyxy[0].cpu().numpy()
                    )

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

                    object_crop = image[
                        y1:y2,
                        x1:x2
                    ]

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

                    category_counts[category] += 1

                    confidence = float(
                        box.conf[0].item()
                    )

                    detected_objects.append({
                        "object_number": object_number,
                        "category": category,
                        "class_id": class_id,
                        "confidence": round(
                            confidence, 4
                        ),
                        "bbox": [
                            x1,
                            y1,
                            x2,
                            y2
                        ],
                        "bounding_box": {
                            "x1": x1,
                            "y1": y1,
                            "x2": x2,
                            "y2": y2
                        },
                        "crop_filename": crop_filename,
                        "crop_path": str(crop_path)
                    })

            # Save annotated frame.
            annotated_frame = result.plot()

            annotated_path = (
                annotated_dir / frame_filename
            )

            if not cv2.imwrite(
                str(annotated_path),
                annotated_frame
            ):
                raise IOError(
                    f"Could not save annotated frame: "
                    f"{annotated_path}"
                )

            frame_total = len(detected_objects)

            total_objects += frame_total

            for category, count in category_counts.items():
                total_category_counts[category] += count

            frame_results.append({
                "frame_number": frame_number,
                "frame_filename": frame_filename,
                "timestamp_seconds": timestamp,
                "frame_width": width,
                "frame_height": height,
                "total_objects": frame_total,
                "category_counts": category_counts,
                "objects": detected_objects,
                "annotated_frame_path": str(annotated_path)
            })

        output = {
            "video_id": video_id,
            "total_frames_processed": len(frame_results),
            "total_objects_detected": total_objects,
            "category_counts": total_category_counts,
            "categories_processed": categories,
            "annotated_frames_folder": str(annotated_dir),
            "objects_folder": str(video_object_dir),
            "frames": frame_results
        }

        metadata_path = (
            video_detection_dir
            / "object_detection_metadata.json"
        )

        with metadata_path.open(
            "w",
            encoding="utf-8"
        ) as file:

            json.dump(
                output,
                file,
                indent=4
            )

        output["metadata_path"] = str(metadata_path)

        return output
