
from uuid import UUID

from fastapi import (
    APIRouter,
    UploadFile,
    File,
    HTTPException,
    Request
)
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from app.controllers.image_controller import ImageController
from app.config import settings


router = APIRouter()


class DetectObjectsRequest(BaseModel):
    object_categories: list[str] | None = Field(
        default=None,
        description=(
            "Categories to detect. If omitted, all supported "
            "categories will be detected."
        )
    )


class CompareImagesRequest(BaseModel):
    image_id_1: str
    image_id_2: list[str] = Field(min_length=1)

class CompareObjectsRequest(BaseModel):
    image_id_1: UUID
    image_id_2: list[UUID] = Field(min_length=1)

# --------------------------------------------------
# Upload and process image
# --------------------------------------------------

@router.post("")
async def process_image(
    request: Request,
    file: UploadFile = File(...)
):
    try:
        allowed_types = [
            "image/jpeg",
            "image/png",
            "image/webp"
        ]

        if file.content_type not in allowed_types:
            raise HTTPException(
                status_code=400,
                detail="Only JPG, PNG and WEBP images are allowed"
            )

        file_bytes = await file.read(
            settings.MAX_FILE_SIZE + 1
        )

        if len(file_bytes) > settings.MAX_FILE_SIZE:
            raise HTTPException(
                status_code=413,
                detail="Image file exceeds the maximum allowed size"
            )

        base_url = str(request.base_url).rstrip("/")

        result = ImageController.process_image(
            file_bytes=file_bytes,
            base_url=base_url
        )

        return {
            "success": True,
            "message": "Image processed successfully",
            "data": result
        }

    except HTTPException:
        raise

    except ValueError as error:
        raise HTTPException(
            status_code=400,
            detail=str(error)
        )

    except Exception as error:
        print("Image processing error:", error)

        raise HTTPException(
            status_code=500,
            detail="Something went wrong while processing image"
        )

    finally:
        await file.close()


# --------------------------------------------------
# Get processed image
# --------------------------------------------------

@router.get("/{image_id}/processed")
def get_processed_image(image_id: UUID):

    try:
        image_path = ImageController.get_processed_image_path(
            str(image_id)
        )

        if image_path is None:
            raise HTTPException(
                status_code=404,
                detail="Processed image not found"
            )

        return FileResponse(
            path=image_path,
            media_type="image/jpeg",
            filename=f"{image_id}.jpg"
        )

    except HTTPException:
        raise

    except Exception as error:
        print("Get processed image error:", error)

        raise HTTPException(
            status_code=500,
            detail="Unable to retrieve processed image"
        )


# --------------------------------------------------
# Existing person detection endpoint
# --------------------------------------------------

@router.post("/{image_id}/detect-persons")
def detect_persons(
    image_id: UUID,
    request: Request
):

    try:
        base_url = str(request.base_url).rstrip("/")

        result = ImageController.detect_persons(
            image_id=str(image_id),
            base_url=base_url
        )

        return {
            "success": True,
            "message": "Person detection completed successfully",
            "data": result
        }

    except ValueError as error:
        raise HTTPException(
            status_code=404,
            detail=str(error)
        )

    except Exception as error:
        print("Person detection error:", error)

        raise HTTPException(
            status_code=500,
            detail="Person detection failed"
        )


# --------------------------------------------------
# New general object detection endpoint
# --------------------------------------------------

@router.post("/{image_id}/detect-objects")
def detect_objects(
    image_id: UUID,
    request: Request,
    payload: DetectObjectsRequest
):

    try:
        base_url = str(request.base_url).rstrip("/")

        result = ImageController.detect_objects(
            image_id=str(image_id),
            object_categories=payload.object_categories,
            base_url=base_url
        )

        return {
            "success": True,
            "message": "Object detection completed successfully",
            "data": result
        }

    except ValueError as error:
        raise HTTPException(
            status_code=400,
            detail=str(error)
        )

    except Exception as error:
        print("Object detection error:", error)

        raise HTTPException(
            status_code=500,
            detail="Object detection failed"
        )


# --------------------------------------------------
# Existing OSNet feature extraction
# --------------------------------------------------

@router.post("/{image_id}/extract-features")
def extract_person_features(image_id: UUID):

    try:
        result = ImageController.extract_person_features(
            str(image_id)
        )

        return {
            "success": True,
            "message": "OSNet feature extraction completed successfully",
            "data": result
        }

    except ValueError as error:
        raise HTTPException(
            status_code=404,
            detail=str(error)
        )

    except FileNotFoundError as error:
        raise HTTPException(
            status_code=500,
            detail=str(error)
        )

    except Exception as error:
        print("OSNet feature extraction error:", error)

        raise HTTPException(
            status_code=500,
            detail="OSNet feature extraction failed"
        )


# --------------------------------------------------
# Existing image comparison
# --------------------------------------------------

@router.post("/compare-images")
def compare_images(request: CompareImagesRequest):

    try:
        result = ImageController.compare_images(
            original_image_id=request.image_id_1,
            search_image_ids=request.image_id_2
        )

        return {
            "success": True,
            "message": (
                "Quality-aware independent image matching "
                "completed successfully"
            ),
            "data": result
        }

    except HTTPException:
        raise

    except FileNotFoundError as error:
        raise HTTPException(
            status_code=404,
            detail=str(error)
        )

    except ValueError as error:
        raise HTTPException(
            status_code=400,
            detail=str(error)
        )

    except OSError as error:
        raise HTTPException(
            status_code=400,
            detail=str(error)
        )

    except Exception as error:
        print("Image comparison error:", error)

        raise HTTPException(
            status_code=500,
            detail="Image comparison failed"
        )


# --------------------------------------------------
# General object feature extraction (DINOv2)
# --------------------------------------------------

@router.post("/{image_id}/extract-object-features")
def extract_object_features(image_id: UUID):

    try:
        result = ImageController.extract_object_features(
            str(image_id)
        )

        return {
            "success": True,
            "message": (
                "DINOv2 object feature extraction "
                "completed successfully"
            ),
            "data": result
        }

    except ValueError as error:
        raise HTTPException(
            status_code=404,
            detail=str(error)
        )

    except FileNotFoundError as error:
        raise HTTPException(
            status_code=404,
            detail=str(error)
        )

    except Exception as error:
        print("Object feature extraction error:", error)

        raise HTTPException(
            status_code=500,
            detail="Object feature extraction failed"
        )

@router.post("/compare-objects")
def compare_objects(request: CompareObjectsRequest):
    try:
        result = ImageController.compare_objects(
            original_image_id=str(request.image_id_1),
            search_image_ids=[
                str(image_id) for image_id in request.image_id_2
            ],
            top_k=5,
            threshold=0.75
        )

        return {
            "success": True,
            "message": "Object comparison completed successfully",
            "data": result
        }

    except ValueError as error:
        raise HTTPException(
            status_code=400,
            detail=str(error)
        )

    except FileNotFoundError as error:
        raise HTTPException(
            status_code=404,
            detail=str(error)
        )

    except Exception as error:
        print("Object comparison error:", error)
        raise HTTPException(
            status_code=500,
            detail="Object comparison failed"
        )
