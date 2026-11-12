
import uuid

from pathlib import Path
from uuid import UUID

from fastapi import (
    APIRouter,
    UploadFile,
    File,
    HTTPException
)

from app.services.osnet_service import EMBEDDING_DIR
from app.services.video_frame_service import VideoFrameService
from app.services.video_person_detection_service import (
    VideoPersonDetectionService
)

from app.services.video_osnet_service import (
    VideoOSNetFeatureService
)

from app.services.video_matching_service import (
    VideoMatchingService
)

router = APIRouter()


# --------------------------------------------------
# Directory Configuration
# --------------------------------------------------

UPLOAD_DIR = EMBEDDING_DIR.parent

VIDEO_DIR = UPLOAD_DIR / "videos"

VIDEO_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# --------------------------------------------------
# Video Configuration
# --------------------------------------------------

ALLOWED_VIDEO_EXTENSIONS = {
    ".mp4",
    ".avi",
    ".mov",
    ".mkv"
}

MAX_VIDEO_SIZE = 500 * 1024 * 1024  # 500 MB


# --------------------------------------------------
# Upload Video
# --------------------------------------------------

@router.post("/upload")
async def upload_video(
    file: UploadFile = File(...)
):

    if not file.filename:
        raise HTTPException(
            status_code=400,
            detail="Video filename is required"
        )

    extension = Path(file.filename).suffix.lower()

    if extension not in ALLOWED_VIDEO_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail="Only MP4, AVI, MOV and MKV videos are allowed"
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

                    raise HTTPException(
                        status_code=413,
                        detail="Video size cannot exceed 500 MB"
                    )

                buffer.write(chunk)

        if total_size == 0:

            raise HTTPException(
                status_code=400,
                detail="Uploaded video is empty"
            )

        return {
            "success": True,
            "message": "Video uploaded successfully",
            "data": {
                "video_id": video_id,
                "original_filename": file.filename,
                "stored_filename": video_path.name,
                "file_size_bytes": total_size,
                "file_path": str(video_path)
            }
        }

    except HTTPException:
        video_path.unlink(missing_ok=True)
        raise

    except Exception as error:

        video_path.unlink(missing_ok=True)

        print("Video upload error:", error)

        raise HTTPException(
            status_code=500,
            detail="Something went wrong while uploading video"
        )

    finally:

        await file.close()


# --------------------------------------------------
# Get Video Information
# --------------------------------------------------

@router.get("/{video_id}")
def get_video(
    video_id: UUID
):

    video_id = str(video_id)

    video_files = list(
        VIDEO_DIR.glob(f"{video_id}.*")
    )

    if not video_files:

        raise HTTPException(
            status_code=404,
            detail="Video not found"
        )

    video_path = video_files[0]

    return {
        "success": True,
        "message": "Video found successfully",
        "data": {
            "video_id": video_id,
            "filename": video_path.name,
            "file_size_bytes": video_path.stat().st_size,
            "file_path": str(video_path)
        }
    }


# --------------------------------------------------
# Extract Video Frames
# --------------------------------------------------

@router.post("/{video_id}/extract-frames")
def extract_video_frames(
    video_id: UUID,
    interval_seconds: float = 1
):

    try:

        result = VideoFrameService.extract_frames(
            video_id=str(video_id),
            interval_seconds=interval_seconds
        )

        return {
            "success": True,
            "message": "Video frames extracted successfully",
            "data": result
        }

    except FileNotFoundError as error:

        raise HTTPException(
            status_code=404,
            detail=str(error)
        )

    except (ValueError, IOError) as error:

        raise HTTPException(
            status_code=400,
            detail=str(error)
        )

    except Exception as error:

        print("Video frame extraction error:", error)

        raise HTTPException(
            status_code=500,
            detail="Something went wrong while extracting video frames"
        )


# --------------------------------------------------
# Detect Persons in Video Frames
# --------------------------------------------------

@router.post("/{video_id}/detect-persons")
def detect_video_persons(
    video_id: UUID,
    interval_seconds: float = 1
):

    try:

        result = VideoPersonDetectionService.detect_persons(
            video_id=str(video_id),
            interval_seconds=interval_seconds
        )

        return {
            "success": True,
            "message": "Video person detection completed successfully",
            "data": result
        }

    except FileNotFoundError as error:

        raise HTTPException(
            status_code=404,
            detail=str(error)
        )

    except (ValueError, IOError) as error:

        raise HTTPException(
            status_code=400,
            detail=str(error)
        )

    except Exception as error:

        print("Video person detection error:", error)

        raise HTTPException(
            status_code=500,
            detail="Video person detection failed"
        )


# --------------------------------------------------
# Extract OSNet Features from Video Person Crops
# --------------------------------------------------

@router.post("/{video_id}/extract-features")
def extract_video_features(
    video_id: UUID,
    interval_seconds: float = 1
):

    try:

        result = VideoOSNetFeatureService.extract_features(
            video_id=str(video_id),
            interval_seconds=interval_seconds
        )

        return {
            "success": True,
            "message": "Video OSNet feature extraction completed successfully",
            "data": result
        }

    except FileNotFoundError as error:

        raise HTTPException(
            status_code=404,
            detail=str(error)
        )

    except (ValueError, IOError) as error:

        raise HTTPException(
            status_code=400,
            detail=str(error)
        )

    except Exception as error:

        print("Video feature extraction error:", error)

        raise HTTPException(
            status_code=500,
            detail="Video feature extraction failed"
        )
        
@router.post("/{video_id}/compare-reference")
def compare_video_with_reference(
    video_id: UUID,
    reference_image_id: UUID,
    interval_seconds: float = 1,
    threshold: float = 0.70,
    top_k: int = 3,
    min_score_gap: float = 0.05
):

    try:

        result = VideoMatchingService.compare_with_reference(
            video_id=str(video_id),
            reference_image_id=str(reference_image_id),
            interval_seconds=interval_seconds,
            threshold=threshold,
            top_k=top_k,
            min_score_gap=min_score_gap
        )

        return {
            "success": True,
            "message": "Video reference matching completed successfully",
            "data": result
        }

    except FileNotFoundError as error:

        raise HTTPException(
            status_code=404,
            detail=str(error)
        )

    except (ValueError, IOError) as error:

        raise HTTPException(
            status_code=400,
            detail=str(error)
        )

    except Exception as error:

        print("Video matching error:", error)

        raise HTTPException(
            status_code=500,
            detail="Video reference matching failed"
        )
