
import numpy as np


class SimilarityService:

    @staticmethod
    def calculate_similarity(embedding_a, embedding_b):

        # Convert input embeddings into NumPy arrays
        vector_a = np.asarray(embedding_a, dtype=np.float32)
        vector_b = np.asarray(embedding_b, dtype=np.float32)

        # Validate dimensions
        if vector_a.ndim != 1 or vector_b.ndim != 1:
            raise ValueError("Embeddings must be one-dimensional vectors")

        # Both embeddings must have the same size
        if vector_a.shape != vector_b.shape:
            raise ValueError("Both embeddings must have the same dimension")

        # Validate numerical values
        if not np.all(np.isfinite(vector_a)) or not np.all(np.isfinite(vector_b)):
            raise ValueError("Embeddings contain invalid numerical values")

        # Calculate vector magnitudes
        norm_a = np.linalg.norm(vector_a)
        norm_b = np.linalg.norm(vector_b)

        if norm_a == 0 or norm_b == 0:
            raise ValueError("Cannot compare zero embeddings")

        # Calculate cosine similarity
        similarity = np.dot(vector_a, vector_b) / (norm_a * norm_b)

        # Handle floating-point rounding errors
        similarity = float(np.clip(similarity, -1.0, 1.0))

        return {
            "similarity_score": round(similarity, 6),
            "similarity_percentage": round(similarity * 100, 2)
        }

    

    @staticmethod
    def rank_matches(
        search_embedding,
        original_persons,
        threshold=0.70,
        top_k=3,
        min_score_gap=0.05
    ):
        matches = []
        # 1. Calculate similarity against every original person
        for person in original_persons:

            result = SimilarityService.calculate_similarity(
                search_embedding,
                person["embedding"]
            )

            matches.append({
                "person_number": person["person_number"],
                "similarity_score": result["similarity_score"],
                "similarity_percentage": result["similarity_percentage"]
            })

        # 2. Sort from highest to lowest similarity
        matches.sort(
            key=lambda item: item["similarity_score"],
            reverse=True
        )

        # 3. Get top candidates
        top_matches = matches[:top_k]

        if not matches:
            return {
                "match_found": False,
                "match_status": "no_candidates",
                "threshold_percentage": threshold * 100,
                "best_similarity": None,
                "second_best_similarity": None,
                "score_gap": None,
                "best_match": None,
                "top_matches": []
            }

        # 4. Get best and second-best scores
        best_match = matches[0]

        best_score = best_match["similarity_score"]

        second_best_score = (
            matches[1]["similarity_score"]
            if len(matches) > 1
            else None
        )

        # 5. Calculate score gap
        score_gap = (
            round(best_score - second_best_score, 6)
            if second_best_score is not None
            else None
        )

        # 6. Apply matching rules
        match_found = best_score >= threshold

        if not match_found:
            match_status = "below_threshold"

        elif score_gap is not None and score_gap < min_score_gap:
            match_status = "ambiguous"

        else:
            match_status = "clear_match"

        return {
            "match_found": match_found,
            "match_status": match_status,
            "threshold_percentage": round(threshold * 100, 2),
            "minimum_score_gap_percentage": round(min_score_gap * 100, 2),
            "best_similarity": round(best_score * 100, 2),
            "second_best_similarity": (
                round(second_best_score * 100, 2)
                if second_best_score is not None
                else None
            ),
            "score_gap": (
                round(score_gap * 100, 2)
                if score_gap is not None
                else None
            ),
            "best_match": best_match if match_found else None,
            "top_matches": top_matches
        }

