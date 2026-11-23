import numpy as np
import torch
from pathlib import Path
from sqlalchemy.orm import Session

from app.config import settings
from app.database.models import Person, PersonEmbedding, PersonZonePermission, Zone
from app.services.osnet_service import OSNetFeatureService
from app.services.image_quality_service import ImageQualityService
from app.services.similarity_service import SimilarityService
from app.services.face_service import (
    extract_face_embedding_from_array,
    find_best_face_match,
)


# Thresholds
SIMILARITY_THRESHOLD = settings.BODY_MATCH_THRESHOLD   # below this = UNKNOWN
MIN_SCORE_GAP = 0.05          # gap between top 2 persons must be > this, else VERIFY


class IdentifyService:

    @staticmethod
    def identify(photo_path: Path, zone_code: str, db: Session) -> dict:
        """
        Identify a person from a photo and decide their access status for a zone.

        Returns one of 5 states:
          - NO_DECISION  : photo too small or blurry to process
          - UNKNOWN      : no match above threshold (possible intruder)
          - VERIFY       : ambiguous match — two persons look too similar (never grant access)
          - UNAUTHORIZED : matched a known person who is NOT allowed in this zone
          - ALLOWED      : matched a known person who IS allowed in this zone
        """

        # --- Step 1: Check zone exists ---
        zone = db.query(Zone).filter(Zone.code == zone_code.strip().upper()).first()
        if not zone:
            return _make_result(
                status="ERROR",
                color="gray",
                message=f"Zone '{zone_code}' does not exist. Create it first.",
                zone_code=zone_code,
                zone_name=None,
            )

        # --- Step 2: Check image quality ---
        quality = ImageQualityService.validate_crop(photo_path)
        if not quality["can_process"]:
            return _make_result(
                status="NO_DECISION",
                color="gray",
                message=f"Image quality too low: {quality['message']}. Minimum 64x128 pixels required.",
                zone_code=zone.code,
                zone_name=zone.name,
                quality=quality,
            )

        # --- Step 3: Extract OSNet embedding from the query photo ---
        extractor = OSNetFeatureService.get_extractor()
        features = extractor([str(photo_path)])

        if isinstance(features, torch.Tensor):
            features = features.detach().cpu().numpy()

        query_embedding = np.asarray(features)[0]
        norm = np.linalg.norm(query_embedding)

        if not np.isfinite(norm) or norm == 0:
            return _make_result(
                status="NO_DECISION",
                color="gray",
                message="Could not extract a valid embedding from this image.",
                zone_code=zone.code,
                zone_name=zone.name,
            )

        query_embedding = (query_embedding / norm).astype(np.float32)

        # --- Step 4: Load all enrolled embeddings from database ---
        all_embeddings = (
            db.query(PersonEmbedding)
            .join(Person, Person.id == PersonEmbedding.person_id)
            .filter(Person.is_active == True, PersonEmbedding.embedding_type == "osnet")
            .all()
        )

        if not all_embeddings:
            return _make_result(
                status="UNKNOWN",
                color="red",
                message="No persons are enrolled in the system yet.",
                zone_code=zone.code,
                zone_name=zone.name,
                similarity_score=0.0,
            )

        # --- Step 5: keep each person's best-matching photo (cosine; both sides are normalized) ---
        best_per_person = {}
        for emb_record in all_embeddings:
            stored_embedding = np.frombuffer(emb_record.embedding, dtype=np.float32)
            similarity = float(np.dot(query_embedding, stored_embedding))
            if similarity > best_per_person.get(emb_record.person_id, (-2.0, None))[0]:
                best_per_person[emb_record.person_id] = (similarity, stored_embedding)

        # SimilarityService wants a list with "person_number"; map those back to DB ids.
        id_to_index = {}
        gallery = []
        for idx, (person_id, (_, embedding)) in enumerate(best_per_person.items(), start=1):
            id_to_index[idx] = person_id
            gallery.append({"person_number": idx, "embedding": embedding})

        # --- Step 6: Run SimilarityService (untouched existing logic) ---
        rank_result = SimilarityService.rank_matches(
            search_embedding=query_embedding,
            original_persons=gallery,
            threshold=SIMILARITY_THRESHOLD,
            top_k=len(gallery),
            min_score_gap=MIN_SCORE_GAP,
        )

        match_status = rank_result["match_status"]
        best_similarity = rank_result["best_similarity"]  # 0-100 scale

        # --- Step 7: Map match_status to 5-state access decision ---

        if match_status == "below_threshold":
            return _make_result(
                status="UNKNOWN",
                color="red",
                message=f"Unknown person detected in {zone.name}. Security alert!",
                zone_code=zone.code,
                zone_name=zone.name,
                similarity_score=best_similarity,
            )

        if match_status == "ambiguous":
            return _make_result(
                status="VERIFY",
                color="amber",
                message="Match is ambiguous — two persons look too similar. Manual verification required.",
                zone_code=zone.code,
                zone_name=zone.name,
                similarity_score=best_similarity,
            )

        # clear_match — find who matched
        matched_person_id = id_to_index[rank_result["best_match"]["person_number"]]
        return _zone_decision(matched_person_id, zone, db, best_similarity / 100)


def _make_result(
    status: str,
    color: str,
    message: str,
    zone_code: str = None,
    zone_name: str = None,
    person_id: int = None,
    person_name: str = None,
    employee_code: str = None,
    department: str = None,
    similarity_score: float = None,
    quality: dict = None,
) -> dict:
    """Helper to build a consistent identification result dict."""
    return {
        "status": status,
        "color": color,
        "message": message,
        "zone_code": zone_code,
        "zone_name": zone_name,
        "person_id": person_id,
        "person_name": person_name,
        "employee_code": employee_code,
        "department": department,
        "similarity_score": similarity_score,
        "quality_info": quality,
    }


def _zone_decision(matched_person_id: int, zone, db: Session, similarity_score: float) -> dict:
    """
    Given a confirmed match, check zone permission and return the final result dict.
    Shared by both OSNet and ArcFace identify paths.
    """
    matched_person = db.query(Person).filter(Person.id == matched_person_id).first()
    if matched_person is None:  # deleted while we were matching
        return _make_result(
            status="UNKNOWN", color="red",
            message=f"Matched person no longer exists. Treated as unknown in {zone.name}.",
            zone_code=zone.code, zone_name=zone.name,
        )
    permission = (
        db.query(PersonZonePermission)
        .filter(
            PersonZonePermission.person_id == matched_person_id,
            PersonZonePermission.zone_id == zone.id,
        )
        .first()
    )

    if permission:
        return _make_result(
            status="ALLOWED", color="green",
            message=f"{matched_person.name} is authorized for {zone.name}.",
            zone_code=zone.code, zone_name=zone.name,
            person_id=matched_person.id, person_name=matched_person.name,
            employee_code=matched_person.employee_code,
            department=matched_person.department,
            similarity_score=round(similarity_score * 100, 2),
        )
    else:
        return _make_result(
            status="UNAUTHORIZED", color="red",
            message=f"{matched_person.name} is NOT authorized for {zone.name}.",
            zone_code=zone.code, zone_name=zone.name,
            person_id=matched_person.id, person_name=matched_person.name,
            employee_code=matched_person.employee_code,
            department=matched_person.department,
            similarity_score=round(similarity_score * 100, 2),
        )


def identify_by_face(crop_bgr: np.ndarray, zone_code: str, db: Session) -> dict | None:
    """
    Try to identify a person using ArcFace face recognition.

    Args:
        crop_bgr: BGR numpy array of the person crop (any size).
                  The upper 40% is used for face detection automatically.
        zone_code: Zone code string.
        db: SQLAlchemy session.

    Returns:
        Result dict (same format as IdentifyService.identify) if face matched,
        or None if no face was detected or no match found.
    """
    zone = db.query(Zone).filter(Zone.code == zone_code.strip().upper()).first()
    if not zone:
        return None

    # Crop the upper 40% of the bounding box — this is where the face is
    h = crop_bgr.shape[0]
    face_region = crop_bgr[:max(1, int(h * 0.45)), :]

    face_emb = extract_face_embedding_from_array(face_region)
    if face_emb is None:
        # Also try the full crop (person may be close to camera)
        face_emb = extract_face_embedding_from_array(crop_bgr)
    if face_emb is None:
        return None   # No face visible — caller should fall back to OSNet

    # Load all stored ArcFace embeddings
    all_face_embs = (
        db.query(PersonEmbedding)
        .join(Person, Person.id == PersonEmbedding.person_id)
        .filter(
            Person.is_active == True,
            PersonEmbedding.embedding_type == "arcface",
        )
        .all()
    )

    if not all_face_embs:
        return None   # No face embeddings enrolled — fall back to OSNet

    gallery = [
        {
            "person_id": rec.person_id,
            "embedding": np.frombuffer(rec.embedding, dtype=np.float32),
        }
        for rec in all_face_embs
    ]

    match = find_best_face_match(face_emb, gallery)
    if match is not None and match["ambiguous"]:
        return _make_result(
            status="VERIFY", color="amber",
            message="Face matches two people about equally. Manual verification required.",
            zone_code=zone.code, zone_name=zone.name,
            similarity_score=round(match["score"] * 100, 2),
        )
    if match is None:
        # Face detected but no match — this is an UNKNOWN person
        return _make_result(
            status="UNKNOWN", color="red",
            message=f"Unknown face detected in {zone.name}. Security alert!",
            zone_code=zone.code, zone_name=zone.name,
            similarity_score=0.0,
        )

    return _zone_decision(match["person_id"], zone, db, match["score"])


def person_has_face_enrolment(db: Session, person_id: int) -> bool:
    """True if this person has at least one ArcFace embedding (so a face mismatch means something)."""
    return db.query(PersonEmbedding.id).filter(
        PersonEmbedding.person_id == person_id,
        PersonEmbedding.embedding_type == "arcface",
    ).first() is not None
