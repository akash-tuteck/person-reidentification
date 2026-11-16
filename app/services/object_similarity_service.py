
import numpy as np


class ObjectSimilarityService:

    EMBEDDING_DIMENSION = 768

    # ---------------------------------------------------------
    # CALCULATE COSINE SIMILARITY
    # ---------------------------------------------------------

    @staticmethod
    def calculate_similarity(
        embedding_a: np.ndarray,
        embedding_b: np.ndarray
    ) -> float:

        vector_a = np.asarray(embedding_a, dtype=np.float32)
        vector_b = np.asarray(embedding_b, dtype=np.float32)

        # Validate dimensions
        if vector_a.ndim != 1 or vector_b.ndim != 1:
            raise ValueError("Embeddings must be one-dimensional")

        if vector_a.shape != vector_b.shape:
            raise ValueError("Embedding dimensions must match")

        if vector_a.size != ObjectSimilarityService.EMBEDDING_DIMENSION:
            raise ValueError(
                f"Expected {ObjectSimilarityService.EMBEDDING_DIMENSION}-D embeddings"
            )

        # Validate numerical values
        if not np.isfinite(vector_a).all() or not np.isfinite(vector_b).all():
            raise ValueError("Embeddings contain invalid values")

        # Calculate vector norms
        norm_a = np.linalg.norm(vector_a)
        norm_b = np.linalg.norm(vector_b)

        if norm_a == 0 or norm_b == 0:
            raise ValueError("Cannot compare zero-vector embeddings")

        # Cosine similarity
        similarity = np.dot(vector_a, vector_b) / (norm_a * norm_b)

        return round(float(np.clip(similarity, -1.0, 1.0)), 4)

    # ---------------------------------------------------------
    # RANK OBJECT MATCHES
    # ---------------------------------------------------------

    @classmethod
    def rank_matches(
        cls,
        original_embedding: np.ndarray,
        candidates: list[dict],
        top_k: int = 5,
        threshold: float | None = None
    ) -> dict:

        if top_k < 1:
            raise ValueError("top_k must be at least 1")

        if threshold is not None and not -1.0 <= threshold <= 1.0:
            raise ValueError("Threshold must be between -1 and 1")

        ranked_matches = []

        for candidate in candidates:

            similarity = cls.calculate_similarity(
                original_embedding,
                candidate["embedding"]
            )
            match_data = {
                key: value
                for key, value in candidate.items()
                if key != "embedding"
            }

            match_data["similarity"] = similarity
            ranked_matches.append(match_data)

        # Highest similarity first
        ranked_matches.sort(
            key=lambda item: item["similarity"],
            reverse=True
        )

        top_matches = ranked_matches[:top_k]

        best_match = top_matches[0] if top_matches else None

        if best_match is None:
            match_status = "no_candidates"
            match_found = False

        elif threshold is None:
            match_status = "ranked"
            match_found = None

        elif best_match["similarity"] >= threshold:
            match_status = "above_threshold"
            match_found = True

        else:
            match_status = "below_threshold"
            match_found = False

        return {
            "match_found": match_found,
            "match_status": match_status,
            "best_match": best_match,
            "total_candidates": len(ranked_matches),
            "top_matches": top_matches
        }
