import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
import numpy as np

from app.services.text_embedding_service import (
    TextEmbeddingService
)


IMAGE_ID = "3f55c3b2-3493-4d4d-9d22-3cf08cc4538a"


if __name__ == "__main__":

    print("\nTEXT EMBEDDING SERVICE TEST")
    print("===========================")

    result = TextEmbeddingService.extract_features(
        IMAGE_ID
    )

    print("\nRESULT")
    print("===========================")

    print(result)

    if result["total_semantic_embeddings"] > 0:

        embedding_path = (
            "app/uploads/embeddings/semantic_objects/"
            f"{IMAGE_ID}/dog/dog_1.npy"
        )

        vector = np.load(embedding_path)

        print("\nVECTOR VALIDATION")
        print("===========================")

        print("Shape:", vector.shape)
        print("Data type:", vector.dtype)
        print("Vector norm:", np.linalg.norm(vector))
        print("All finite:", np.isfinite(vector).all())
