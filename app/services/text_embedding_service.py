
import logging
import numpy as np
import torch

from pathlib import Path
from sentence_transformers import SentenceTransformer


logger = logging.getLogger(__name__)


BASE_DIR = Path(__file__).resolve().parent.parent

UPLOAD_DIR = BASE_DIR / "uploads"

DESCRIPTION_DIR = UPLOAD_DIR / "object_descriptions"

SEMANTIC_EMBEDDING_DIR = (
    UPLOAD_DIR / "embeddings" / "semantic_objects"
)

SEMANTIC_EMBEDDING_DIR.mkdir(
    parents=True,
    exist_ok=True
)


class TextEmbeddingService:

    MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

    EMBEDDING_DIMENSION = 384

    _model = None

    _device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    @classmethod
    def get_model(cls):
        """
        Load the sentence embedding model only once.
        """

        if cls._model is None:

            cls._model = SentenceTransformer(
                cls.MODEL_NAME,
                device=str(cls._device)
            )

        return cls._model

    @classmethod
    def build_text(cls, metadata: dict) -> str:
        """
        Convert Gemma structured metadata into descriptive text.
        """

        parts = []

        category = metadata.get("category")
        description = metadata.get("description")

        if category:
            parts.append(f"Object category: {category}")

        if description:
            parts.append(f"Description: {description}")

        attributes = [
            ("Color", metadata.get("color")),
            ("Shape", metadata.get("shape")),
            (
                "Surface or material",
                metadata.get("surface_or_material")
            ),
            ("Condition", metadata.get("condition"))
        ]

        for label, value in attributes:

            if value:
                parts.append(f"{label}: {value}")

        characteristics = metadata.get(
            "visible_characteristics",
            []
        )

        if isinstance(characteristics, list) and characteristics:

            parts.append(
                "Visible characteristics: "
                + ", ".join(
                    str(item) for item in characteristics
                )
            )

        text = ". ".join(parts)

        if not text.strip():
            raise ValueError(
                "No valid text available for semantic embedding"
            )

        return text

    @classmethod
    def extract_embedding(cls, text: str) -> np.ndarray:
        """
        Convert text into a normalized 384-dimensional vector.
        """

        if not isinstance(text, str) or not text.strip():
            raise ValueError("Input text cannot be empty")

        model = cls.get_model()

        embedding = model.encode(
            text,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False
        )

        vector = np.asarray(
            embedding,
            dtype=np.float32
        )

        if vector.shape != (cls.EMBEDDING_DIMENSION,):
            raise ValueError(
                f"Unexpected semantic embedding dimension: {vector.shape}"
            )

        if not np.isfinite(vector).all():
            raise ValueError(
                "Semantic embedding contains invalid values"
            )

        return vector

    @classmethod
    def extract_features(cls, image_id: str) -> dict:
        """
        Generate semantic embeddings for all successful
        Gemma descriptions belonging to an image.
        """

        image_description_dir = DESCRIPTION_DIR / image_id

        if not image_description_dir.exists():
            raise ValueError(
                f"Object descriptions not found: {image_id}"
            )

        image_embedding_dir = (
            SEMANTIC_EMBEDDING_DIR / image_id
        )

        image_embedding_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        total_processed = 0
        total_skipped = 0

        category_counts = {}
        processed_objects = []

        for category_dir in sorted(image_description_dir.iterdir()):

            if not category_dir.is_dir():
                continue

            category = category_dir.name

            json_files = sorted(
                category_dir.glob("*.json")
            )

            if not json_files:
                continue

            category_embedding_dir = (
                image_embedding_dir / category
            )

            category_embedding_dir.mkdir(
                parents=True,
                exist_ok=True
            )

            category_count = 0

            for json_path in json_files:

                try:

                    import json

                    with json_path.open(
                        "r",
                        encoding="utf-8"
                    ) as file:

                        saved_metadata = json.load(file)

                    if saved_metadata.get("status") != "completed":
                        total_skipped += 1
                        continue

                    metadata = saved_metadata.get("description")

                    if not isinstance(metadata, dict):
                        raise ValueError(
                            "Invalid description metadata"
                        )

                    text = cls.build_text(metadata)

                    vector = cls.extract_embedding(text)

                    embedding_path = (
                        category_embedding_dir /
                        f"{json_path.stem}.npy"
                    )

                    np.save(
                        embedding_path,
                        vector
                    )

                    category_count += 1
                    total_processed += 1

                    processed_objects.append({
                        "category": category,
                        "object_filename": json_path.stem,
                        "embedding_filename": embedding_path.name,
                        "embedding_dimension": int(vector.shape[0])
                    })

                except Exception:

                    logger.exception(
                        "Semantic embedding failed for %s",
                        json_path
                    )

                    total_skipped += 1

            if category_count > 0:
                category_counts[category] = category_count

        return {
            "image_id": image_id,
            "total_semantic_embeddings": total_processed,
            "total_skipped": total_skipped,
            "category_counts": category_counts,
            "embedding_dimension": cls.EMBEDDING_DIMENSION,
            "model": cls.MODEL_NAME,
            "device": str(cls._device),
            "status": (
                "completed" if total_processed > 0
                else "no_embeddings_generated"
            ),
            "processed_objects": processed_objects
        }

