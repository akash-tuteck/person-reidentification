import numpy as np
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.services.similarity_service import SimilarityService


def create_embedding(similarity):
    """
    Creates a normalized 512-dimensional vector whose
    cosine similarity with [1, 0, 0, ...] is the given value.
    """
    return np.array(
        [
            similarity,
            np.sqrt(1 - similarity ** 2),
            *([0.0] * 510)
        ],
        dtype=np.float32
    )


search_embedding = create_embedding(1.0)

test_cases = [
    {
        "name": "Clear match",
        "scores": [0.95, 0.60],
        "expected": "clear_match"
    },
    {
        "name": "Below threshold",
        "scores": [0.55, 0.40],
        "expected": "below_threshold"
    },
    {
        "name": "Ambiguous match",
        "scores": [0.90, 0.88],
        "expected": "ambiguous"
    }
]

for case in test_cases:

    original_persons = [
        {
            "person_number": index + 1,
            "embedding": create_embedding(score)
        }
        for index, score in enumerate(case["scores"])
    ]

    result = SimilarityService.rank_matches(
        search_embedding=search_embedding,
        original_persons=original_persons,
        threshold=0.70,
        top_k=3,
        min_score_gap=0.05
    )

    print("\nScenario:", case["name"])
    print("Expected:", case["expected"])
    print("Actual:", result["match_status"])
    print("Best similarity:", result["best_similarity"])
    print("Second similarity:", result["second_best_similarity"])
    print("Score gap:", result["score_gap"])

    assert result["match_status"] == case["expected"], (
        f"Test failed: {case['name']}"
    )

print("\nAll matching scenario tests passed!")