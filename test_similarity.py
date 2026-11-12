
import numpy as np

from pathlib import Path
from app.services.similarity_service import SimilarityService


BASE_DIR = Path(__file__).resolve().parent

# Group photograph
SEARCH_IMAGE_ID = "e009c5aa-b4e9-48a8-bfe2-3b9e10651b1c"

# Single-person reference photograph
REFERENCE_IMAGE_ID = "0c2a24ce-0311-411c-9535-4b19d997cd1f"

REFERENCE_PERSON_NUMBER = 1


def get_embedding_path(image_id):

    folder = (
        BASE_DIR
        / "app"
        / "uploads"
        / "embeddings"
        / image_id
    )

    if not folder.exists():
        raise FileNotFoundError(
            f"Embedding folder not found: {folder}"
        )

    return folder


def load_embedding(image_id, person_number):

    path = (
        get_embedding_path(image_id)
        / f"person_{person_number}.npy"
    )

    if not path.exists():
        raise FileNotFoundError(
            f"Embedding not found: {path}"
        )

    return np.load(path)


# Load reference person
reference_embedding = load_embedding(
    REFERENCE_IMAGE_ID,
    REFERENCE_PERSON_NUMBER
)

# Find all candidate embeddings in the group image
search_folder = get_embedding_path(SEARCH_IMAGE_ID)

candidate_files = sorted(
    search_folder.glob("person_*.npy"),
    key=lambda path: int(path.stem.split("_")[1])
)

if not candidate_files:
    raise ValueError("No candidate embeddings found")


# Compare reference against every candidate
results = []

for candidate_file in candidate_files:

    person_number = int(
        candidate_file.stem.split("_")[1]
    )

    candidate_embedding = np.load(candidate_file)

    similarity = SimilarityService.calculate_similarity(
        reference_embedding,
        candidate_embedding
    )

    results.append({
        "person_number": person_number,
        **similarity
    })


# Sort by highest similarity
results.sort(
    key=lambda item: item["similarity_score"],
    reverse=True
)


print("\nREFERENCE PERSON SEARCH RESULTS")
print("--------------------------------")

for rank, result in enumerate(results, start=1):

    print(
        f"Rank: {rank} | "
        f"Person: {result['person_number']} | "
        f"Similarity: {result['similarity_percentage']}%"
    )
