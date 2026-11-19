
from uuid import UUID

from fastapi import (
    APIRouter,
    UploadFile,
    File,
    HTTPException
)

from app.controllers.video_controller import VideoController


router = APIRouter()


# --------------------------------------------------
# UPLOAD VIDEO
# --------------------------------------------------

@router.post("/upload")
async def upload_video(
    file: UploadFile = File(...)
):

    try:

        result = await VideoController.upload_video(file)

        return {
            "success": True,
            "message": "Video uploaded successfully",
            "data": result
        }

    except OverflowError as error:

        raise HTTPException(
            status_code=413,
            detail=str(error)
        )

    except ValueError as error:

        raise HTTPException(
            status_code=400,
            detail=str(error)
        )

    except Exception as error:

        print("Video upload error:", error)

        raise HTTPException(
            status_code=500,
            detail="Something went wrong while uploading video"
        )


# --------------------------------------------------
# GET VIDEO INFORMATION
# --------------------------------------------------

@router.get("/{video_id}")
def get_video(video_id: UUID):

    try:

        result = VideoController.get_video(
            video_id=str(video_id)
        )

        return {
            "success": True,
            "message": "Video found successfully",
            "data": result
        }

    except FileNotFoundError as error:

        raise HTTPException(
            status_code=404,
            detail=str(error)
        )


# --------------------------------------------------
# EXTRACT VIDEO FRAMES
# --------------------------------------------------

@router.post("/{video_id}/extract-frames")
def extract_video_frames(
    video_id: UUID,
    interval_seconds: float = 1
):

    try:

        result = VideoController.extract_frames(
            video_id=str(video_id),
            interval_seconds=interval_seconds
        )

        return {
            "success": True,
            "message": "Video frames extracted successfully",
            "data": result
        }

    except FileNotFoundError as error:
        raise HTTPException(status_code=404, detail=str(error))

    except (ValueError, IOError) as error:
        raise HTTPException(status_code=400, detail=str(error))

    except Exception as error:

        print("Video frame extraction error:", error)

        raise HTTPException(
            status_code=500,
            detail="Something went wrong while extracting video frames"
        )


# --------------------------------------------------
# DETECT PERSONS IN VIDEO
# --------------------------------------------------

@router.post("/{video_id}/detect-persons")
def detect_video_persons(
    video_id: UUID,
    interval_seconds: float = 1
):

    try:

        result = VideoController.detect_persons(
            video_id=str(video_id),
            interval_seconds=interval_seconds
        )

        return {
            "success": True,
            "message": "Video person detection completed successfully",
            "data": result
        }

    except FileNotFoundError as error:
        raise HTTPException(status_code=404, detail=str(error))

    except (ValueError, IOError) as error:
        raise HTTPException(status_code=400, detail=str(error))

    except Exception as error:

        print("Video person detection error:", error)

        raise HTTPException(
            status_code=500,
            detail="Video person detection failed"
        )


# --------------------------------------------------
# EXTRACT OSNET FEATURES
# --------------------------------------------------

@router.post("/{video_id}/extract-features")
def extract_video_features(
    video_id: UUID,
    interval_seconds: float = 1
):

    try:

        result = VideoController.extract_features(
            video_id=str(video_id),
            interval_seconds=interval_seconds
        )

        return {
            "success": True,
            "message": "Video OSNet feature extraction completed successfully",
            "data": result
        }

    except FileNotFoundError as error:
        raise HTTPException(status_code=404, detail=str(error))

    except (ValueError, IOError) as error:
        raise HTTPException(status_code=400, detail=str(error))

    except Exception as error:

        print("Video feature extraction error:", error)

        raise HTTPException(
            status_code=500,
            detail="Video feature extraction failed"
        )


# --------------------------------------------------
# TRACK PERSONS
# --------------------------------------------------

@router.post("/{video_id}/track-persons")
def track_video_persons(video_id: UUID):

    try:

        result = VideoController.track_persons(
            video_id=str(video_id)
        )

        return {
            "success": True,
            "message": "Video person tracking completed successfully",
            "data": result
        }

    except FileNotFoundError as error:
        raise HTTPException(status_code=404, detail=str(error))

    except (ValueError, IOError) as error:
        raise HTTPException(status_code=400, detail=str(error))

    except Exception as error:

        print("Video person tracking error:", error)

        raise HTTPException(
            status_code=500,
            detail="Video person tracking failed"
        )


# --------------------------------------------------
# COMPARE VIDEO WITH REFERENCE IMAGE
# --------------------------------------------------

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

        result = VideoController.compare_reference(
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
        raise HTTPException(status_code=404, detail=str(error))

    except (ValueError, IOError) as error:
        raise HTTPException(status_code=400, detail=str(error))

    except Exception as error:

        print("Video matching error:", error)

        raise HTTPException(
            status_code=500,
            detail="Video reference matching failed"
        )


# --------------------------------------------------
# DETECT NON-HUMAN OBJECTS
# --------------------------------------------------

@router.post("/{video_id}/detect-objects")
def detect_video_objects(video_id: UUID):

    try:

        result = VideoController.detect_objects(
            video_id=str(video_id)
        )

        return {
            "success": True,
            "message": "Video object detection completed successfully",
            "data": result
        }

    except FileNotFoundError as error:
        raise HTTPException(status_code=404, detail=str(error))

    except (ValueError, IOError) as error:
        raise HTTPException(status_code=400, detail=str(error))

    except Exception as error:

        print("Video object detection error:", error)

        raise HTTPException(
            status_code=500,
            detail="Video object detection failed"
        )


# --------------------------------------------------
# MATCH VIDEO PERSONS WITH REGISTERED EMPLOYEES
# --------------------------------------------------

@router.post("/{video_id}/match-employees")
def match_video_employees(video_id: UUID):

    try:

        result = VideoController.match_employees(
            video_id=str(video_id)
        )

        return {
            "success": True,
            "message": "Video employee matching completed successfully",
            "data": result
        }

    except FileNotFoundError as error:
        raise HTTPException(status_code=404, detail=str(error))

    except (ValueError, IOError) as error:
        raise HTTPException(status_code=400, detail=str(error))

    except Exception as error:

        print("Video employee matching error:", error)

        raise HTTPException(
            status_code=500,
            detail="Video employee matching failed"
        )
