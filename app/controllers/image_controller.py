
import numpy as np

from pathlib import Path

from app.services.image_service import ImageProcessingService
from app.services.person_detection_service import PersonDetectionService
from app.services.object_detection_service import ObjectDetectionService
from app.services.osnet_service import OSNetFeatureService, EMBEDDING_DIR
from app.services.image_quality_service import ImageQualityService
from app.services.similarity_service import SimilarityService
from app.services.object_feature_service import ObjectFeatureService
from app.services.object_similarity_service import ObjectSimilarityService
from app.services.grounding_dino_service import GroundingDINOService

class ImageController:
    PERSON_DIR = EMBEDDING_DIR.parent / "persons"
    @staticmethod
    def process_image(file_bytes: bytes, base_url: str) -> dict:

        result = ImageProcessingService.process_image(file_bytes)

        image_id = result["image_id"]

        original_path = Path(result["original_path"])
        original_extension = original_path.suffix

        return {
            "image_id": image_id,
            "original_width": result["original_width"],
            "original_height": result["original_height"],
            "processed_width": result["processed_width"],
            "processed_height": result["processed_height"],
            "original_image_url": (
                f"{base_url}/uploads/original/"
                f"{image_id}{original_extension}"
            ),
            "processed_image_url": (
                f"{base_url}/uploads/processed/{image_id}.jpg"
            )
        }

    # ---------------------------------------------------------
    # GET PROCESSED IMAGE
    # ---------------------------------------------------------

    @staticmethod
    def get_processed_image_path(image_id: str):

        image_path = ImageProcessingService.get_processed_image_path(
            image_id
        )

        return image_path

    # ---------------------------------------------------------
    # PERSON DETECTION
    # ---------------------------------------------------------

    @staticmethod
    def detect_persons(image_id: str, base_url: str) -> dict:

        result = PersonDetectionService.detect_persons(image_id)

        persons = []

        for person in result["persons"]:

            persons.append({
                **person,
                "crop_image_url": (
                    f"{base_url}/uploads/persons/"
                    f"{image_id}/{person['crop_filename']}"
                )
            })

        return {
            "image_id": result["image_id"],
            "total_persons": result["total_persons"],
            "annotated_image_url": (
                f"{base_url}/uploads/detections/{image_id}.jpg"
            ),
            "persons": persons
        }

    # ---------------------------------------------------------
    # GENERAL OBJECT DETECTION
    # ---------------------------------------------------------

    @staticmethod
    def detect_objects(
        image_id: str,
        object_categories: list[str] | None,
        base_url: str
    ) -> dict:

        result = ObjectDetectionService.detect_objects(
            image_id=image_id,
            object_categories=object_categories
        )

        objects = []

        for obj in result["objects"]:

            object_data = {
                key: value
                for key, value in obj.items()
                if key != "crop_path"
            }

            object_data["crop_image_url"] = (
                f"{base_url}/uploads/objects/"
                f"{image_id}/{obj['category']}/{obj['crop_filename']}"
            )

            objects.append(object_data)

        return {
            "image_id": result["image_id"],
            "total_objects": result["total_objects"],
            "category_counts": result["category_counts"],
            "annotated_image_url": (
                f"{base_url}/uploads/detections/{image_id}.jpg"
            ),
            "objects": objects
        }


    # ---------------------------------------------------------
    # AUTOMATIC OBJECT DISCOVERY + GROUNDING DINO
    # ---------------------------------------------------------
    @staticmethod
    def detect_objects_automatically(
        image_id: str,
        base_url: str
    ) -> dict:
        result = GroundingDINOService.detect_objects_automatically(
            image_id=image_id
        )
        detection = result.get("detection")
        if detection is not None:
            objects = []
            for item in detection.get("objects", []):
                object_data = {
                    key: value
                    for key, value in item.items()
                    if key != "crop_path"
                }
                object_data["crop_image_url"] = (
                    f"{base_url}/uploads/grounding_dino/objects/"
                    f"{image_id}/{item['category']}/{item['crop_filename']}"
                )
                objects.append(object_data)
            detection["objects"] = objects
            detection["annotated_image_url"] = (
                f"{base_url}/uploads/grounding_dino/detections/{image_id}.jpg"
            )
            detection.pop("annotated_image_path", None)
            detection.pop("metadata_path", None)
        return result

    # ---------------------------------------------------------
    # PERSON FEATURE EXTRACTION
    # ---------------------------------------------------------

    @staticmethod
    def extract_person_features(image_id: str) -> dict:

        result = OSNetFeatureService.extract_features(image_id)

        return result

    # ---------------------------------------------------------
    # PERSON IMAGE QUALITY
    # ---------------------------------------------------------

    @classmethod
    def get_person_quality(
        cls,
        image_id: str,
        person_number: int
    ) -> dict:

        crop_path = (
            cls.PERSON_DIR
            / image_id
            / f"person_{person_number}.jpg"
        )

        if not crop_path.exists():
            raise ValueError(
                f"Person crop not found: {crop_path}"
            )

        return ImageQualityService.validate_crop(
            str(crop_path)
        )

    # ---------------------------------------------------------
    # IMAGE COMPARISON
    # ---------------------------------------------------------

    @classmethod
    def compare_images(
        cls,
        original_image_id: str,
        search_image_ids: list[str]
    ) -> dict:

        original_embedding_dir = EMBEDDING_DIR / original_image_id

        if not original_embedding_dir.exists():
            raise ValueError(
                f"Original image embeddings not found: "
                f"{original_image_id}"
            )

        original_embedding_files = sorted(
            original_embedding_dir.glob("person_*.npy")
        )

        if not original_embedding_files:
            raise ValueError(
                "No person embeddings found for the original image"
            )

        results = []

        total_original_persons = len(original_embedding_files)

        for search_image_id in search_image_ids:

            search_embedding_dir = EMBEDDING_DIR / search_image_id

            if not search_embedding_dir.exists():
                results.append({
                    "search_image_id": search_image_id,
                    "success": False,
                    "message": "Search image embeddings not found",
                    "matches": []
                })
                continue

            search_embedding_files = sorted(
                search_embedding_dir.glob("person_*.npy")
            )

            if not search_embedding_files:
                results.append({
                    "search_image_id": search_image_id,
                    "success": False,
                    "message": "No person embeddings found",
                    "matches": []
                })
                continue

            search_embeddings = []

            for embedding_file in search_embedding_files:

                embedding = np.load(embedding_file)

                search_embeddings.append({
                    "person_number": int(
                        embedding_file.stem.split("_")[-1]
                    ),
                    "embedding": embedding
                })

            original_person_matches = []

            for original_file in original_embedding_files:

                original_person_number = int(
                    original_file.stem.split("_")[-1]
                )

                original_embedding = np.load(original_file)

                candidates = [
                    {
                        "person_number": item["person_number"],
                        "embedding": item["embedding"]
                    }
                    for item in search_embeddings
                ]

                ranked_matches = SimilarityService.rank_matches(
                    original_embedding,
                    candidates,
                    threshold=0.70,
                    top_k=3
                )

                original_quality = cls.get_person_quality(
                    original_image_id,
                    original_person_number
                )

                enriched_matches = []

                for match in ranked_matches.get("top_matches", []):

                    search_person_number = match["person_number"]

                    search_quality = cls.get_person_quality(
                        search_image_id,
                        search_person_number
                    )

                    enriched_matches.append({
                        **match,
                        "quality": search_quality
                    })

                original_person_matches.append({
                    "original_person_number": original_person_number,
                    "original_quality": original_quality,
                    "match_found": ranked_matches.get(
                        "match_found",
                        False
                    ),
                    "match_status": ranked_matches.get(
                        "match_status"
                    ),
                    "best_match": ranked_matches.get("best_match"),
                    "top_matches": enriched_matches
                })

            results.append({
                "search_image_id": search_image_id,
                "success": True,
                "total_search_persons": len(search_embeddings),
                "matches": original_person_matches
            })

        return {
            "original_image_id": original_image_id,
            "total_original_persons": total_original_persons,
            "total_search_images": len(search_image_ids),
            "results": results
        }

        # ---------------------------------------------------------
    # GENERAL OBJECT FEATURE EXTRACTION (DINOv2)
    # ---------------------------------------------------------

    @staticmethod
    def extract_object_features(image_id: str) -> dict:

        result = ObjectFeatureService.extract_features(
            image_id=image_id
        )

        return result
    
    @classmethod
    def compare_objects(
        cls,
        original_image_id: str,
        search_image_ids: list[str],
        top_k: int = 5,
        threshold: float | None = None
    ) -> dict:

        object_embedding_dir = Path(EMBEDDING_DIR) / "objects"

        original_dir = object_embedding_dir / original_image_id

        if not original_dir.exists():
            raise ValueError(
                f"Object embeddings not found for image: {original_image_id}"
            )

        original_files = list(original_dir.rglob("*.npy"))

        if not original_files:
            raise ValueError(
                f"No object embeddings found for image: {original_image_id}"
            )

        comparison_results = []
        missing_image_ids = []

        for original_file in original_files:

            category = original_file.parent.name

            original_embedding = np.load(original_file)

            candidates = []

            # Search only the same object category
            for candidate_image_id in search_image_ids:

                candidate_category_dir = (
                    object_embedding_dir
                    / candidate_image_id
                    / category
                )

                if not candidate_category_dir.exists():
                    continue

                candidate_files = list(
                    candidate_category_dir.glob("*.npy")
                )

                for candidate_file in candidate_files:

                    candidate_embedding = np.load(candidate_file)

                    try:
                        object_number = int(
                            candidate_file.stem.rsplit("_", 1)[-1]
                        )
                    except ValueError:
                        object_number = candidate_file.stem

                    candidates.append({
                        "image_id": candidate_image_id,
                        "object_number": object_number,
                        "category": category,
                        "object_filename": candidate_file.stem + ".jpg",
                        "embedding": candidate_embedding
                    })

            # Rank candidate objects
            match_result = ObjectSimilarityService.rank_matches(
                original_embedding=original_embedding,
                candidates=candidates,
                top_k=top_k,
                threshold=threshold
            )

            try:
                original_object_number = int(
                    original_file.stem.rsplit("_", 1)[-1]
                )
            except ValueError:
                original_object_number = original_file.stem

            comparison_results.append({
                "original_object": {
                    "image_id": original_image_id,
                    "category": category,
                    "object_number": original_object_number,
                    "object_filename": original_file.stem + ".jpg"
                },
                "comparison": match_result
            })

        for candidate_image_id in search_image_ids:
            if not (object_embedding_dir / candidate_image_id).exists():
                missing_image_ids.append(candidate_image_id)

        return {
            "original_image_id": original_image_id,
            "searched_image_ids": search_image_ids,
            "total_original_objects": len(original_files),
            "total_comparisons": len(comparison_results),
            "missing_image_ids": missing_image_ids,
            "results": comparison_results
        }

