
import uuid

from pathlib import Path
from fastapi import UploadFile, HTTPException

from app.services.osnet_service import EMBEDDING_DIR
from app.services.video_frame_service import VideoFrameService
from app.services.video_person_detection_service import (
    VideoPersonDetectionService
)
from app.services.video_tracking_service import VideoTrackingService
from app.services.video_osnet_service import (
    VideoOSNetFeatureService
)
from app.services.video_matching_service import VideoMatchingService
from app.services.video_object_detection_service import (
    VideoObjectDetectionService
)
from app.services.video_employee_matching_service import (
    VideoEmployeeMatchingService
)


UPLOAD_DIR = EMBEDDING_DIR.parent
VIDEO_DIR = UPLOAD_DIR / "videos"

VIDEO_DIR.mkdir(parents=True, exist_ok=True)

ALLOWED_VIDEO_EXTENSIONS = {
    ".mp4",
    ".avi",
    ".mov",
    ".mkv"
}

MAX_VIDEO_SIZE = 500 * 1024 * 1024


class VideoController:

    # --------------------------------------------------
    # UPLOAD VIDEO
    # --------------------------------------------------

    @staticmethod
    async def upload_video(file: UploadFile) -> dict:

        if not file.filename:
            raise ValueError("Video filename is required")

        extension = Path(file.filename).suffix.lower()

        if extension not in ALLOWED_VIDEO_EXTENSIONS:
            raise ValueError(
                "Only MP4, AVI, MOV and MKV videos are allowed"
            )

        video_id = str(uuid.uuid4())
        video_path = VIDEO_DIR / f"{video_id}{extension}"

        total_size = 0

        try:
            with video_path.open("wb") as buffer:

                while True:

                    chunk = await file.read(1024 * 1024)

                    if not chunk:
                        break

                    total_size += len(chunk)

                    if total_size > MAX_VIDEO_SIZE:
                        raise OverflowError(
                            "Video size cannot exceed 500 MB"
                        )

                    buffer.write(chunk)

            if total_size == 0:
                raise ValueError("Uploaded video is empty")

            return {
                "video_id": video_id,
                "original_filename": file.filename,
                "stored_filename": video_path.name,
                "file_size_bytes": total_size,
                "file_path": str(video_path)
            }

        except Exception:
            video_path.unlink(missing_ok=True)
            raise

        finally:
            await file.close()

    # --------------------------------------------------
    # GET VIDEO INFORMATION
    # --------------------------------------------------

    @staticmethod
    def get_video(video_id: str) -> dict:

        video_files = list(
            VIDEO_DIR.glob(f"{video_id}.*")
        )

        if not video_files:
            raise FileNotFoundError("Video not found")

        video_path = video_files[0]

        return {
            "video_id": video_id,
            "filename": video_path.name,
            "file_size_bytes": video_path.stat().st_size,
            "file_path": str(video_path)
        }

    # --------------------------------------------------
    # EXTRACT VIDEO FRAMES
    # --------------------------------------------------

    @staticmethod
    def extract_frames(
        video_id: str,
        interval_seconds: float = 1
    ) -> dict:

        return VideoFrameService.extract_frames(
            video_id=video_id,
            interval_seconds=interval_seconds
        )

    # --------------------------------------------------
    # DETECT PERSONS
    # --------------------------------------------------

    @staticmethod
    def detect_persons(
        video_id: str,
        interval_seconds: float = 1
    ) -> dict:

        return VideoPersonDetectionService.detect_persons(
            video_id=video_id,
            interval_seconds=interval_seconds
        )

    # --------------------------------------------------
    # EXTRACT OSNET FEATURES
    # --------------------------------------------------

    @staticmethod
    def extract_features(
        video_id: str,
        interval_seconds: float = 1
    ) -> dict:

        return VideoOSNetFeatureService.extract_features(
            video_id=video_id,
            interval_seconds=interval_seconds
        )

    # --------------------------------------------------
    # TRACK PERSONS
    # --------------------------------------------------

    @staticmethod
    def track_persons(video_id: str) -> dict:

        return VideoTrackingService.track_persons(
            video_id=video_id
        )

    # --------------------------------------------------
    # COMPARE VIDEO WITH REFERENCE IMAGE
    # --------------------------------------------------

    @staticmethod
    def compare_reference(
        video_id: str,
        reference_image_id: str,
        interval_seconds: float = 1,
        threshold: float = 0.70,
        top_k: int = 3,
        min_score_gap: float = 0.05
    ) -> dict:

        return VideoMatchingService.compare_with_reference(
            video_id=video_id,
            reference_image_id=reference_image_id,
            interval_seconds=interval_seconds,
            threshold=threshold,
            top_k=top_k,
            min_score_gap=min_score_gap
        )

    # --------------------------------------------------
    # DETECT NON-HUMAN OBJECTS
    # --------------------------------------------------

    @staticmethod
    def detect_objects(video_id: str) -> dict:

        return VideoObjectDetectionService.detect_objects(
            video_id=video_id
        )

    # --------------------------------------------------
    # MATCH VIDEO PERSONS WITH REGISTERED EMPLOYEES
    # --------------------------------------------------

    @staticmethod
    def match_employees(video_id: str) -> dict:

        return VideoEmployeeMatchingService.match_video(
            video_id=video_id
        )
