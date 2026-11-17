
import cv2
import json
import shutil

from pathlib import Path

from app.services.osnet_service import EMBEDDING_DIR
from app.services.person_detection_service import PersonDetectionService


UPLOAD_DIR = EMBEDDING_DIR.parent

FRAME_DIR = UPLOAD_DIR / "frames"
VIDEO_PERSON_DIR = UPLOAD_DIR / "video_persons"
VIDEO_DETECTION_DIR = UPLOAD_DIR / "video_detections"


class VideoPersonDetectionService:

    @staticmethod
    def detect_persons(video_id, interval_seconds=1):

        if interval_seconds <= 0:
            raise ValueError("Interval must be greater than zero")

        frame_folder = FRAME_DIR / str(video_id)

        if not frame_folder.exists():
            raise FileNotFoundError(
                f"Extracted frames not found for video: {video_id}"
            )

        # Load frame metadata
        metadata_path = frame_folder / "frames_metadata.json"

        if not metadata_path.exists():
            raise FileNotFoundError(
                "Frame metadata not found. Extract frames again first."
            )

        with open(metadata_path, "r", encoding="utf-8") as file:
            frame_metadata = json.load(file)

        # Use metadata as the source of truth.
        # This prevents old or duplicate JPG files from being processed.
        frame_records = frame_metadata.get("frames", [])

        if not frame_records:
            raise ValueError("No extracted frames found in metadata")

        model = PersonDetectionService.get_model()

        video_person_folder = VIDEO_PERSON_DIR / str(video_id)
        video_detection_folder = VIDEO_DETECTION_DIR / str(video_id)

        # Remove previous outputs completely.
        # This also removes old frame folders such as frame_000001_at_0ms.
        for folder in (video_person_folder, video_detection_folder):
            if folder.exists():
                shutil.rmtree(folder)

        video_person_folder.mkdir(parents=True, exist_ok=True)
        video_detection_folder.mkdir(parents=True, exist_ok=True)

        total_detected_persons = 0
        total_frames_processed = 0
        total_frames_with_persons = 0
        total_frames_without_persons = 0

        skipped_frames = []
        frame_results = []

        for frame_info in frame_records:

            frame_filename = frame_info["frame_filename"]
            frame_path = frame_folder / frame_filename

            if not frame_path.exists():
                skipped_frames.append({
                    "frame_filename": frame_filename,
                    "reason": "Frame file not found"
                })
                continue

            frame = cv2.imread(str(frame_path))

            if frame is None:
                skipped_frames.append({
                    "frame_filename": frame_filename,
                    "reason": "Unable to read frame"
                })
                continue

            frame_number = frame_info["frame_number"]
            timestamp_seconds = frame_info["timestamp_seconds"]

            height, width = frame.shape[:2]

            results = model.predict(
                source=frame,
                classes=[0],
                conf=0.35,
                imgsz=640,
                verbose=False
            )

            result = results[0]
            annotated_frame = result.plot()

            annotated_path = video_detection_folder / frame_filename

            if not cv2.imwrite(str(annotated_path), annotated_frame):
                raise IOError(
                    f"Unable to save annotated frame: {frame_filename}"
                )

            frame_person_folder = (
                video_person_folder / frame_path.stem
            )

            frame_person_folder.mkdir(
                parents=True,
                exist_ok=True
            )

            detected_persons = []

            boxes = result.boxes

            if boxes is not None:

                for index, box in enumerate(boxes):

                    coordinates = (
                        box.xyxy[0]
                        .cpu()
                        .numpy()
                        .astype(int)
                    )

                    x1, y1, x2, y2 = coordinates

                    x1 = max(0, min(x1, width))
                    y1 = max(0, min(y1, height))
                    x2 = max(0, min(x2, width))
                    y2 = max(0, min(y2, height))

                    if x2 <= x1 or y2 <= y1:
                        continue

                    person_crop = frame[y1:y2, x1:x2]

                    if person_crop.size == 0:
                        continue

                    person_number = index + 1

                    crop_filename = f"person_{person_number}.jpg"
                    crop_path = frame_person_folder / crop_filename

                    if not cv2.imwrite(str(crop_path), person_crop):
                        continue

                    confidence = float(box.conf[0].cpu().item())

                    detected_persons.append({
                        "person_number": person_number,
                        "confidence": round(confidence, 4),
                        "bounding_box": {
                            "x1": int(x1),
                            "y1": int(y1),
                            "x2": int(x2),
                            "y2": int(y2)
                        },
                        "crop_filename": crop_filename,
                        "crop_path": str(crop_path)
                    })

            person_count = len(detected_persons)

            total_detected_persons += person_count

            if person_count:
                total_frames_with_persons += 1
            else:
                total_frames_without_persons += 1

            total_frames_processed += 1

            frame_result = {
                "frame_number": frame_number,
                "frame_filename": frame_filename,
                "timestamp_seconds": timestamp_seconds,
                "frame_width": width,
                "frame_height": height,
                "total_persons": person_count,
                "annotated_frame_path": str(annotated_path),
                "persons": detected_persons
            }

            frame_results.append(frame_result)

            detection_json_path = (
                video_detection_folder
                / f"{frame_path.stem}_metadata.json"
            )

            with open(
                detection_json_path,
                "w",
                encoding="utf-8"
            ) as file:
                json.dump(frame_result, file, indent=4)

        complete_metadata = {
            "video_id": str(video_id),
            "total_frames": len(frame_records),
            "total_frames_processed": total_frames_processed,
            "total_frames_with_persons": total_frames_with_persons,
            "total_frames_without_persons": total_frames_without_persons,
            "total_detected_persons": total_detected_persons,
            "total_skipped_frames": len(skipped_frames),
            "frames": frame_results,
            "skipped_frames": skipped_frames
        }

        summary_path = (
            video_detection_folder / "detection_metadata.json"
        )

        with open(summary_path, "w", encoding="utf-8") as file:
            json.dump(complete_metadata, file, indent=4)

        return {
            **complete_metadata,
            "video_person_folder": str(video_person_folder),
            "video_detection_folder": str(video_detection_folder),
            "metadata_path": str(summary_path)
        }
