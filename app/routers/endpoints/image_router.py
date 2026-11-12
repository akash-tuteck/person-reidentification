
from uuid import UUID

import numpy as np

from fastapi import APIRouter, UploadFile, File, HTTPException, Request
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from app.services.person_detection_service import PersonDetectionService
from app.services.osnet_service import OSNetFeatureService, EMBEDDING_DIR
from app.services.image_service import ImageProcessingService
from app.services.image_quality_service import ImageQualityService
from app.services.similarity_service import SimilarityService
from app.config import settings


class CompareImagesRequest(BaseModel):
    image_id_1: str
    image_id_2: list[str] = Field(min_length=1)


router = APIRouter()


# Person crop directory
PERSON_DIR = EMBEDDING_DIR.parent / "persons"


# Check individual person image quality
def get_person_quality(image_id, person_number):

    person_path = (
        PERSON_DIR
        / image_id
        / f"person_{person_number}.jpg"
    )

    if not person_path.exists():
        return {
            "available": False,
            "quality_status": "unavailable",
            "message": "Person crop not found"
        }

    quality = ImageQualityService.validate_crop(person_path)

    return {
        "available": True,
        **quality
    }


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

        file_bytes = await file.read(settings.MAX_FILE_SIZE + 1)

        result = ImageProcessingService.process_image(file_bytes)

        base_url = str(request.base_url).rstrip("/")
        image_id = result["image_id"]

        return {
            "success": True,
            "message": "Image processed successfully",
            "data": {
                "image_id": image_id,
                "original_width": result["original_width"],
                "original_height": result["original_height"],
                "processed_width": result["processed_width"],
                "processed_height": result["processed_height"],
                "original_image_url": (
                    f"{base_url}/uploads/original/{image_id}."
                    f"{result['original_path'].rsplit('.', 1)[-1]}"
                ),
                "processed_image_url": (
                    f"{base_url}/uploads/processed/{image_id}.jpg"
                )
            }
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


@router.get("/{image_id}/processed")
def get_processed_image(image_id: UUID):

    image_path = ImageProcessingService.get_processed_image_path(
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


@router.post("/{image_id}/detect-persons")
def detect_persons(image_id: UUID):

    try:
        result = PersonDetectionService.detect_persons(
            str(image_id)
        )

        base_url = "http://127.0.0.1:8000"

        return {
            "success": True,
            "message": "Person detection completed successfully",
            "data": {
                "image_id": result["image_id"],
                "total_persons": result["total_persons"],
                "annotated_image_url": (
                    f"{base_url}/uploads/detections/{image_id}.jpg"
                ),
                "persons": [
                    {
                        **person,
                        "crop_image_url": (
                            f"{base_url}/uploads/persons/"
                            f"{image_id}/{person['crop_filename']}"
                        )
                    }
                    for person in result["persons"]
                ]
            }
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


@router.post("/{image_id}/extract-features")
def extract_person_features(image_id: UUID):

    try:
        result = OSNetFeatureService.extract_features(str(image_id))

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


@router.post("/compare-images")
def compare_images(request: CompareImagesRequest):

    try:

        # 1. Load original image embeddings
        original_folder = EMBEDDING_DIR / request.image_id_1

        if not original_folder.exists():
            raise HTTPException(
                status_code=404,
                detail="Original image embeddings not found"
            )

        original_files = sorted(
            original_folder.glob("person_*.npy"),
            key=lambda p: int(p.stem.split("_")[1])
        )

        if not original_files:
            raise HTTPException(
                status_code=404,
                detail="No people found in original image"
            )

        original_people = []
        original_quality = {}

        # 2. Load original embeddings and quality information
        for file in original_files:

            person_number = int(file.stem.split("_")[1])

            embedding = np.load(
                file,
                allow_pickle=False
            )

            quality = get_person_quality(
                request.image_id_1,
                person_number
            )

            original_quality[person_number] = quality

            original_people.append({
                "person_number": person_number,
                "embedding": embedding
            })

        # 3. Process each search image independently
        final_results = []

        for search_image_id in request.image_id_2:

            search_folder = EMBEDDING_DIR / search_image_id

            if not search_folder.exists():
                raise HTTPException(
                    status_code=404,
                    detail=f"Embeddings not found for image: {search_image_id}"
                )

            search_files = sorted(
                search_folder.glob("person_*.npy"),
                key=lambda p: int(p.stem.split("_")[1])
            )

            if not search_files:
                raise HTTPException(
                    status_code=404,
                    detail=f"No person embeddings found for image: {search_image_id}"
                )

            image_results = []

            # 4. Compare each search person against original people
            for search_file in search_files:

                search_person_number = int(
                    search_file.stem.split("_")[1]
                )

                search_embedding = np.load(
                    search_file,
                    allow_pickle=False
                )

                # 5. Check search person quality
                search_quality = get_person_quality(
                    search_image_id,
                    search_person_number
                )

                # 6. Perform existing independent matching
                matching_result = SimilarityService.rank_matches(
                    search_embedding=search_embedding,
                    original_persons=original_people,
                    threshold=0.70,
                    top_k=3
                )

                # 7. Add quality information to top matches
                enriched_top_matches = []

                for match in matching_result["top_matches"]:

                    original_person_number = match["person_number"]

                    candidate_quality = original_quality.get(
                        original_person_number,
                        {
                            "available": False,
                            "quality_status": "unavailable"
                        }
                    )

                    enriched_top_matches.append({
                        **match,
                        "original_person_quality": candidate_quality
                    })

                # 8. Add quality information to best match
                best_match = matching_result.get("best_match")

                if best_match is not None:

                    best_person_number = best_match["person_number"]

                    best_match = {
                        **best_match,
                        "original_person_quality": original_quality.get(
                            best_person_number,
                            {
                                "available": False,
                                "quality_status": "unavailable"
                            }
                        )
                    }

                # 9. Generate quality warnings
                quality_warnings = []

                if (
                    not search_quality.get("available")
                    or search_quality.get("quality_status") != "good"
                ):
                    quality_warnings.append(
                        "Search person image quality may affect match reliability."
                    )

                # Check quality of the highest-ranked original candidate,
                # even when the similarity threshold was not reached.
                if matching_result["top_matches"]:

                    highest_candidate_number = (
                        matching_result["top_matches"][0]["person_number"]
                    )

                    highest_candidate_quality = original_quality.get(
                        highest_candidate_number,
                        {}
                    )

                    if (
                        not highest_candidate_quality.get("available")
                        or highest_candidate_quality.get("quality_status") != "good"
                    ):
                        quality_warnings.append(
                            "Highest-ranked original candidate image quality may affect match reliability."
                        )

                # 10. Build search person result
                image_results.append({
                    **matching_result,
                    "search_person_number": search_person_number,
                    "search_person_quality": search_quality,
                    "best_match": best_match,
                    "top_matches": enriched_top_matches,
                    "quality_warnings": quality_warnings
                })

            # 11. Store results for this search image
            final_results.append({
                "search_image_id": search_image_id,
                "total_search_persons": len(search_files),
                "person_results": image_results
            })

        # 12. Final API response
        return {
            "success": True,
            "message": "Quality-aware independent image matching completed successfully",
            "data": {
                "original_image_id": request.image_id_1,
                "total_original_persons": len(original_people),
                "total_search_images": len(request.image_id_2),
                "results": final_results
            }
        }

    except HTTPException:
        raise

    except (ValueError, OSError) as error:
        raise HTTPException(
            status_code=400,
            detail=str(error)
        )

    except Exception as error:
        print("Image comparison error:", error)

        raise HTTPException(
            status_code=500,
            detail=str(error)
        )
