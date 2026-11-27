
import json
import logging
import re

import numpy as np
import torch

from pathlib import Path
from PIL import Image
from transformers import AutoImageProcessor, AutoModel

from app.services.gemma_service import GemmaService


logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent

UPLOAD_DIR = BASE_DIR / "uploads"
LEGACY_OBJECT_DIR = UPLOAD_DIR / "objects"
GROUNDING_DINO_OBJECT_DIR = UPLOAD_DIR / "grounding_dino" / "objects"
EMBEDDING_DIR = UPLOAD_DIR / "embeddings"

OBJECT_EMBEDDING_DIR = EMBEDDING_DIR / "objects"
OBJECT_DESCRIPTION_DIR = UPLOAD_DIR / "object_descriptions"

OBJECT_EMBEDDING_DIR.mkdir(parents=True, exist_ok=True)
OBJECT_DESCRIPTION_DIR.mkdir(parents=True, exist_ok=True)


class ObjectFeatureService:
    MODEL_NAME = "facebook/dinov2-base"
    EMBEDDING_DIMENSION = 768

    EXCLUDED_CATEGORIES = {
        "person",
        "people",
        "human",
        "humans"
    }

    CATEGORY_PATTERN = re.compile(
        r"^[a-z0-9][a-z0-9_-]{0,63}$"
    )

    ALLOWED_IMAGE_EXTENSIONS = {
        ".jpg",
        ".jpeg",
        ".png"
    }

    _model = None
    _processor = None
    _device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    @classmethod
    def get_model(cls):

        if cls._model is None:

            cls._processor = AutoImageProcessor.from_pretrained(
                cls.MODEL_NAME
            )

            cls._model = AutoModel.from_pretrained(
                cls.MODEL_NAME
            )

            cls._model.to(cls._device)
            cls._model.eval()

        return cls._model

    @classmethod
    def get_detected_categories(cls, image_object_dir: Path):

        categories = []

        for category_dir in sorted(image_object_dir.iterdir()):

            if not category_dir.is_dir():
                continue

            category = category_dir.name.lower()

            if category in cls.EXCLUDED_CATEGORIES:
                logger.warning(
                    "Skipping person category: %s",
                    category
                )
                continue

            if not cls.CATEGORY_PATTERN.fullmatch(category):
                logger.warning(
                    "Skipping invalid category directory: %s",
                    category
                )
                continue

            categories.append(category)

        return categories

    @classmethod
    def extract_embedding(cls, image_path: Path) -> np.ndarray:

        if not image_path.exists():
            raise FileNotFoundError(
                f"Object crop not found: {image_path}"
            )

        with Image.open(image_path) as source_image:
            image = source_image.convert("RGB")

        model = cls.get_model()

        inputs = cls._processor(
            images=image,
            return_tensors="pt"
        )

        inputs = {
            key: value.to(cls._device)
            for key, value in inputs.items()
        }

        with torch.inference_mode():

            outputs = model(**inputs)

            embedding = outputs.last_hidden_state[:, 0, :]

            embedding = torch.nn.functional.normalize(
                embedding,
                p=2,
                dim=1
            )

        vector = embedding.squeeze(0).cpu().numpy().astype(
            np.float32
        )

        if vector.shape != (cls.EMBEDDING_DIMENSION,):
            raise ValueError(
                f"Unexpected embedding dimension: {vector.shape}"
            )

        if not np.isfinite(vector).all():
            raise ValueError(
                "Embedding contains invalid values"
            )

        return vector

    @classmethod
    def generate_object_description(
        cls,
        crop_path: Path,
        category: str
    ) -> dict:

        return GemmaService.describe_object(
            image_path=crop_path,
            detected_category=category
        )

    @classmethod
    def save_description(
        cls,
        image_id: str,
        category: str,
        crop_path: Path,
        description_path: Path
    ) -> dict:

        try:

            result = cls.generate_object_description(
                crop_path=crop_path,
                category=category
            )

            if not result.get("success"):
                raise ValueError(
                    "Gemma did not return a successful result"
                )

            metadata = {
                "image_id": image_id,
                "category": category,
                "object_filename": crop_path.name,
                "model": result.get("model", "Gemma 4"),
                "description": result["data"],
                "status": "completed"
            }

        except Exception as exc:

            logger.exception(
                "Gemma description failed for %s",
                crop_path
            )

            metadata = {
                "image_id": image_id,
                "category": category,
                "object_filename": crop_path.name,
                "model": "Gemma 4",
                "description": None,
                "status": "failed",
                "error": str(exc)
            }

        description_path.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        with description_path.open(
            "w",
            encoding="utf-8"
        ) as file:

            json.dump(
                metadata,
                file,
                indent=4,
                ensure_ascii=False
            )

        return metadata

    @classmethod
    def extract_features(cls, image_id: str) -> dict:

        # Prefer Grounding DINO crops; retain compatibility with legacy YOLO crops.
        grounding_dino_dir = GROUNDING_DINO_OBJECT_DIR / image_id
        legacy_object_dir = LEGACY_OBJECT_DIR / image_id
        image_object_dir = (
            grounding_dino_dir
            if grounding_dino_dir.exists()
            else legacy_object_dir
        )

        if not image_object_dir.exists():
            raise ValueError(
                f"Object detections not found for image: {image_id}. "
                "Run object detection first."
            )

        embedding_image_dir = OBJECT_EMBEDDING_DIR / image_id
        description_image_dir = OBJECT_DESCRIPTION_DIR / image_id

        embedding_image_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        description_image_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        total_processed = 0
        failed_embedding_count = 0
        gemma_success_count = 0
        gemma_failure_count = 0

        category_counts = {}
        processed_objects = []
        failed_objects = []

        categories = cls.get_detected_categories(
            image_object_dir
        )

        for category in categories:

            category_dir = image_object_dir / category

            crop_files = sorted(
                crop_path
                for crop_path in category_dir.iterdir()
                if crop_path.is_file()
                and crop_path.suffix.lower()
                in cls.ALLOWED_IMAGE_EXTENSIONS
            )

            if not crop_files:
                continue

            category_embedding_dir = (
                embedding_image_dir / category
            )

            category_embedding_dir.mkdir(
                parents=True,
                exist_ok=True
            )

            category_description_dir = (
                description_image_dir / category
            )

            category_description_dir.mkdir(
                parents=True,
                exist_ok=True
            )

            category_count = 0

            for crop_path in crop_files:

                try:

                    # STEP 1: DINOv2 visual embedding.
                    embedding = cls.extract_embedding(
                        crop_path
                    )

                    embedding_path = (
                        category_embedding_dir /
                        f"{crop_path.stem}.npy"
                    )

                    np.save(
                        embedding_path,
                        embedding
                    )

                except Exception as exc:

                    failed_embedding_count += 1

                    logger.exception(
                        "Visual feature extraction failed: %s",
                        crop_path
                    )

                    failed_objects.append({
                        "category": category,
                        "object_filename": crop_path.name,
                        "stage": "visual_embedding",
                        "error": str(exc)
                    })

                    continue

                # STEP 2: Gemma 4 structured description.
                description_path = (
                    category_description_dir /
                    f"{crop_path.stem}.json"
                )

                description_metadata = cls.save_description(
                    image_id=image_id,
                    category=category,
                    crop_path=crop_path,
                    description_path=description_path
                )

                if description_metadata["status"] == "completed":
                    gemma_success_count += 1
                else:
                    gemma_failure_count += 1

                category_count += 1
                total_processed += 1

                processed_objects.append({
                    "category": category,
                    "object_filename": crop_path.name,
                    "embedding_filename": embedding_path.name,
                    "embedding_dimension": int(
                        embedding.shape[0]
                    ),
                    "description_filename": description_path.name,
                    "description_status": description_metadata["status"]
                })

            if category_count > 0:
                category_counts[category] = category_count

        return {
            "image_id": image_id,
            "total_objects_processed": total_processed,
            "category_counts": category_counts,
            "embedding_dimension": cls.EMBEDDING_DIMENSION,
            "visual_model": cls.MODEL_NAME,
            "description_model": "Gemma 4",
            "device": str(cls._device),
            "gemma_success_count": gemma_success_count,
            "gemma_failure_count": gemma_failure_count,
            "failed_embedding_count": failed_embedding_count,
            "failed_objects": failed_objects,
            "status": (
                "completed" if total_processed > 0
                else "no_objects_found"
            ),
            "processed_objects": processed_objects
        }
