import numpy as np
from pathlib import Path
import threading
import cv2


# ── Singleton model holder ────────────────────────────────────────────────────
_app = None
_app_lock = threading.Lock()

# ArcFace threshold — face embeddings are very discriminative.
# 0.45 is the standard InsightFace threshold for buffalo_l on CPU.
FACE_SIMILARITY_THRESHOLD = 0.45
FACE_DET_SCORE_MIN        = 0.50   # Lowered from 0.65 — CCTV faces are often at angle/distance


def _get_app():
    """Return the InsightFace app (loaded once, shared across threads)."""
    global _app
    with _app_lock:
        if _app is None:
            import insightface
            from insightface.app import FaceAnalysis
            # buffalo_l = RetinaFace detector + ArcFace R100 recogniser (best accuracy)
            _app = FaceAnalysis(
                name="buffalo_l",
                providers=["CPUExecutionProvider"],
            )
            # det_size controls how large the input image is for face detection.
            # 320x320 is fast on CPU while still detecting faces in CCTV footage.
            _app.prepare(ctx_id=0, det_size=(320, 320))
            print("[FaceService] InsightFace buffalo_l loaded (CPU)")
    return _app


def extract_face_embedding(image_path: Path) -> np.ndarray | None:
    """
    Detect the largest / best face in an image and return its ArcFace embedding.

    Args:
        image_path: Path to any image (JPG/PNG).  Can be a clear portrait or CCTV crop.

    Returns:
        512-D float32 L2-normalised embedding, or None if no face detected.
    """
    img = cv2.imread(str(image_path))
    if img is None:
        return None

    return extract_face_embedding_from_array(img)


def extract_face_embedding_from_array(img: np.ndarray) -> np.ndarray | None:
    """
    Same as extract_face_embedding but accepts a NumPy BGR array directly.
    Used by the stream worker to avoid writing a temp file.
    """
    app = _get_app()

    faces = app.get(img)
    if not faces:
        return None

    # Pick the face with the highest detection score (most confident)
    best_face = max(faces, key=lambda f: f.det_score)

    if best_face.det_score < FACE_DET_SCORE_MIN:
        return None   # Not confident enough

    # ArcFace embedding is already L2-normalised inside InsightFace
    embedding = np.array(best_face.embedding, dtype=np.float32)
    norm = np.linalg.norm(embedding)
    if norm == 0:
        return None
    return embedding / norm


def face_similarity(emb_a: np.ndarray, emb_b: np.ndarray) -> float:
    """Cosine similarity between two L2-normalised embeddings (range -1 to 1)."""
    return float(np.dot(emb_a, emb_b))


FACE_MIN_SCORE_GAP = 0.05  # two different people this close, both above threshold -> ambiguous


def find_best_face_match(
    query_embedding: np.ndarray,
    gallery: list[dict],  # [{"person_id": int, "embedding": np.ndarray}, ...]
) -> dict | None:
    """
    Compare query against all stored ArcFace embeddings (best photo per person).

    Returns:
        {"person_id": int, "score": float, "ambiguous": bool} for the best person above the
        threshold, or None. ambiguous=True when a second person scores almost the same.
    """
    best_per_person: dict[int, float] = {}
    for entry in gallery:
        score = face_similarity(query_embedding, entry["embedding"])
        if score > best_per_person.get(entry["person_id"], -2.0):
            best_per_person[entry["person_id"]] = score

    ranked = sorted(best_per_person.items(), key=lambda item: item[1], reverse=True)
    if not ranked or ranked[0][1] < FACE_SIMILARITY_THRESHOLD:
        return None

    best_id, best_score = ranked[0]
    ambiguous = (
        len(ranked) > 1
        and ranked[1][1] >= FACE_SIMILARITY_THRESHOLD
        and best_score - ranked[1][1] < FACE_MIN_SCORE_GAP
    )
    return {"person_id": best_id, "score": best_score, "ambiguous": ambiguous}
