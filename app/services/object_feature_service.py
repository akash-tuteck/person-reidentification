
import numpy as np
import torch

from pathlib import Path
from PIL import Image
from transformers import AutoImageProcessor, AutoModel  # type: ignore[reportMissingImports]

from app.services.object_categories import (
    NON_PERSON_COCO_CLASS_IDS
)

BASE_DIR = Path(__file__).resolve().parent.parent

UPLOAD_DIR = BASE_DIR / "uploads"
OBJECT_DIR = UPLOAD_DIR / "objects"
EMBEDDING_DIR = UPLOAD_DIR / "embeddings"

OBJECT_EMBEDDING_DIR = EMBEDDING_DIR / "objects"

OBJECT_EMBEDDING_DIR.mkdir(parents=True, exist_ok=True)


class ObjectFeatureService:
    MODEL_NAME = "facebook/dinov2-base"
    EMBEDDING_DIMENSION = 768

    SUPPORTED_CATEGORIES = set(
        NON_PERSON_COCO_CLASS_IDS.keys()
    )

    _model = None
    _processor = None
    _device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    @classmethod
    def get_model(cls):
        """
        Load DINOv2 only once.
        """

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
    def extract_embedding(cls, image_path: Path) -> np.ndarray:
        """
        Extract a normalized 768-dimensional embedding.
        """

        if not image_path.exists():
            raise FileNotFoundError(
                f"Object crop not found: {image_path}"
            )

        image = Image.open(image_path).convert("RGB")

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
            raise ValueError("Embedding contains invalid values")

        return vector

    @classmethod
    def extract_features(cls, image_id: str) -> dict:
        """
        Extract features for all detected non-person object crops.
        """

        image_object_dir = OBJECT_DIR / image_id

        if not image_object_dir.exists():
            raise ValueError(
                f"Object detections not found for image: {image_id}"
            )

        embedding_image_dir = OBJECT_EMBEDDING_DIR / image_id
        embedding_image_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        total_processed = 0
        category_counts = {}
        processed_objects = []

        for category in sorted(cls.SUPPORTED_CATEGORIES):

            category_dir = image_object_dir / category

            if not category_dir.exists():
                continue

            crop_files = sorted(
                category_dir.glob("*.jpg")
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

            category_count = 0

            for crop_path in crop_files:

                embedding = cls.extract_embedding(crop_path)

                embedding_path = (
                    category_embedding_dir /
                    f"{crop_path.stem}.npy"
                )

                np.save(
                    embedding_path,
                    embedding
                )

                category_count += 1
                total_processed += 1

                processed_objects.append({
                    "category": category,
                    "object_filename": crop_path.name,
                    "embedding_filename": embedding_path.name,
                    "embedding_dimension": int(
                        embedding.shape[0]
                    )
                })

            category_counts[category] = category_count

        return {
            "image_id": image_id,
            "total_objects_processed": total_processed,
            "category_counts": category_counts,
            "embedding_dimension": cls.EMBEDDING_DIMENSION,
            "model": cls.MODEL_NAME,
            "device": str(cls._device),
            "status": (
                "completed" if total_processed > 0
                else "no_objects_found"
            ),
            "processed_objects": processed_objects
        }
