
import json
import shutil
import re

import cv2

from pathlib import Path

from app.services.video_service import VideoProcessingService
from app.services.object_detection_service import ObjectDetectionService


BASE_DIR = Path(__file__).resolve().parent.parent

UPLOAD_DIR = BASE_DIR / "uploads"

VIDEO_DETECTION_DIR = UPLOAD_DIR / "video_detections"
VIDEO_PERSON_DIR = UPLOAD_DIR / "video_persons"

VIDEO_DETECTION_DIR.mkdir(parents=True, exist_ok=True)
VIDEO_PERSON_DIR.mkdir(parents=True, exist_ok=True)


class VideoDetectionService:

    CONFIDENCE_THRESHOLD = 0.35
    IMAGE_SIZE = 640

    @staticmethod
    def extract_frame_metadata(frame_filename: str):

        frame_match = re.search(
            r"frame_(\d+)_at_(\d+)ms",
            frame_filename
        )

        if not frame_match:
            return {
                "frame_number": None,
                "timestamp_ms": None
            }

        return {
            "frame_number": int(frame_match.group(1)),
            "timestamp_ms": int(frame_match.group(2))
        }

    @classmethod
    def detect_video_frames(
        cls,
        video_id: str,
        object_categories: list[str] | None = None
    ):

        # Get extracted frames
        frames_directory = (
            VideoProcessingService.get_frames_directory(video_id)
        )

        if frames_directory is None:
            raise ValueError(
                f"Extracted frames not found for video: {video_id}"
            )

        frame_paths = sorted(
            frames_directory.glob("*.jpg")
        )

        if not frame_paths:
            raise ValueError(
                "No extracted frames found for this video"
            )

        # Use existing YOLO category configuration
        if object_categories is None:
            categories = list(
                ObjectDetectionService.CATEGORY_CLASS_IDS.keys()
            )
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
            if category not in
            ObjectDetectionService.CATEGORY_CLASS_IDS
        ]

        if invalid_categories:
            raise ValueError(
                f"Unsupported categories: {invalid_categories}"
            )

        class_ids = [
            ObjectDetectionService.CATEGORY_CLASS_IDS[category]
            for category in categories
        ]

        # Reuse the existing YOLO model
        model = ObjectDetectionService.get_model()

        frame_results = []

        total_persons = 0
        total_objects = 0

        for frame_path in frame_paths:

            frame = cv2.imread(str(frame_path))

            if frame is None:
                raise ValueError(
                    f"Unable to read frame: {frame_path.name}"
                )

            height, width = frame.shape[:2]

            metadata = cls.extract_frame_metadata(
                frame_path.name
            )

            frame_number = metadata["frame_number"]
            timestamp_ms = metadata["timestamp_ms"]

            frame_name = frame_path.stem

            # Separate output directories for this frame
            detection_frame_dir = (
                VIDEO_DETECTION_DIR
                / video_id
                / frame_name
            )

            person_frame_dir = (
                VIDEO_PERSON_DIR
                / video_id
                / frame_name
            )

            # Remove previous results for this frame
            if detection_frame_dir.exists():
                shutil.rmtree(detection_frame_dir)

            if person_frame_dir.exists():
                shutil.rmtree(person_frame_dir)

            detection_frame_dir.mkdir(
                parents=True,
                exist_ok=True
            )

            person_frame_dir.mkdir(
                parents=True,
                exist_ok=True
            )

            # Run YOLO detection
            results = model.predict(
                source=str(frame_path),
                classes=class_ids,
                conf=cls.CONFIDENCE_THRESHOLD,
                imgsz=cls.IMAGE_SIZE,
                verbose=False
            )

            result = results[0]

            annotated_path = (
                detection_frame_dir / "annotated.jpg"
            )

            if not cv2.imwrite(
                str(annotated_path),
                result.plot()
            ):
                raise RuntimeError(
                    f"Unable to save annotated frame: {frame_path.name}"
                )

            category_counts = {
                category: 0
                for category in categories
            }

            detected_objects = []
            detected_persons = []

            if result.boxes is not None:

                for box in result.boxes:

                    class_id = int(box.cls[0].item())
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

                    crop = frame[y1:y2, x1:x2]

                    category_counts[category] += 1

                    object_number = category_counts[category]

                    crop_filename = (
                        f"{category}_{object_number}.jpg"
                    )

                    # Save people separately
                    if category == "person":

                        crop_path = (
                            person_frame_dir / crop_filename
                        )

                    else:

                        category_dir = (
                            detection_frame_dir / category
                        )

                        category_dir.mkdir(
                            parents=True,
                            exist_ok=True
                        )

                        crop_path = (
                            category_dir / crop_filename
                        )

                    if not cv2.imwrite(
                        str(crop_path),
                        crop
                    ):
                        continue

                    confidence = round(
                        float(box.conf[0].item()),
                        4
                    )

                    detection_data = {
                        "object_number": object_number,
                        "category": category,
                        "class_id": class_id,
                        "confidence": confidence,
                        "bounding_box": {
                            "x1": x1,
                            "y1": y1,
                            "x2": x2,
                            "y2": y2
                        },
                        "crop_filename": crop_filename,
                        "crop_path": str(crop_path)
                    }

                    detected_objects.append(
                        detection_data
                    )

                    if category == "person":

                        detected_persons.append(
                            detection_data
                        )

            total_persons += len(detected_persons)
            total_objects += len(detected_objects)

            # Save frame detection metadata
            frame_result = {
                "video_id": video_id,
                "frame_number": frame_number,
                "timestamp_ms": timestamp_ms,
                "frame_filename": frame_path.name,
                "total_objects": len(detected_objects),
                "total_persons": len(detected_persons),
                "category_counts": category_counts,
                "annotated_image_path": str(annotated_path),
                "objects": detected_objects,
                "persons": detected_persons
            }

            metadata_path = (
                detection_frame_dir / "detection_results.json"
            )

            with open(
                metadata_path,
                "w",
                encoding="utf-8"
            ) as metadata_file:

                json.dump(
                    frame_result,
                    metadata_file,
                    indent=4
                )

            frame_results.append(frame_result)

        return {
            "video_id": video_id,
            "total_frames_processed": len(frame_results),
            "total_objects_detected": total_objects,
            "total_persons_detected": total_persons,
            "categories_processed": categories,
            "results": frame_results,
            "status": "completed"
        }
