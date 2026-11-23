"""
Access-decision tests: no models, no network, no camera. Run: python -m test.test_access_decisions
(also collected by pytest). Uses an in-memory SQLite database and fake embeddings.
"""
import os
import sys
from pathlib import Path
from unittest.mock import patch

os.environ.setdefault("API_KEY", "test-key")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.database import models as m
from app.database.session import Base
from app.services import identify_service, stream_service
from app.services.face_service import find_best_face_match
from app.services.identify_service import IdentifyService, person_has_face_enrolment

DIM = 8


def unit(*values):
    v = np.zeros(DIM, dtype=np.float32)
    v[: len(values)] = values
    return v / np.linalg.norm(v)


def make_db():
    engine = create_engine("sqlite://", poolclass=StaticPool, connect_args={"check_same_thread": False})
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine, autoflush=False)


def add_person(db, name, zone_ids, embeddings):
    """embeddings: list of (type, vector)."""
    person = m.Person(name=name)
    db.add(person)
    db.flush()
    for type_, vec in embeddings:
        db.add(m.PersonEmbedding(person_id=person.id, embedding=vec.tobytes(), embedding_type=type_))
    for zone_id in zone_ids:
        db.add(m.PersonZonePermission(person_id=person.id, zone_id=zone_id))
    db.flush()
    return person


def identify_with(db, query_vec, zone_code="LAB"):
    """Run IdentifyService.identify with a fake OSNet that returns query_vec."""
    with patch.object(identify_service.OSNetFeatureService, "get_extractor", return_value=lambda paths: [query_vec]), \
         patch.object(identify_service.ImageQualityService, "validate_crop", return_value={"can_process": True}):
        return IdentifyService.identify(Path("fake.jpg"), zone_code, db)


def seeded():
    Session = make_db()
    db = Session()
    lab = m.Zone(code="LAB", name="Lab")
    vault = m.Zone(code="VAULT", name="Vault")
    db.add_all([lab, vault])
    db.flush()
    return db, Session, lab, vault


def test_allowed_and_unauthorized():
    db, _, lab, vault = seeded()
    add_person(db, "Asha", [lab.id], [("osnet", unit(1, 0.1))])
    query = unit(1, 0.15)  # very close to Asha
    assert identify_with(db, query, "LAB")["status"] == "ALLOWED"
    result = identify_with(db, query, "VAULT")
    assert result["status"] == "UNAUTHORIZED" and result["person_name"] == "Asha"


def test_unknown_when_nobody_is_close():
    db, _, lab, _ = seeded()
    add_person(db, "Asha", [lab.id], [("osnet", unit(1, 0))])
    assert identify_with(db, unit(0, 1))["status"] == "UNKNOWN"


def test_body_match_ignores_face_embeddings():
    # Bob's ArcFace row is identical to the query. Before the fix it was compared as if it were OSNet.
    db, _, lab, _ = seeded()
    add_person(db, "Asha", [lab.id], [("osnet", unit(1, 0.1))])
    query = unit(1, 0.4)
    add_person(db, "Bob", [], [("arcface", query)])
    result = identify_with(db, query)
    assert result["person_name"] == "Asha" and result["status"] == "ALLOWED", result


def test_face_match_threshold_and_ambiguity():
    a, b = unit(1, 0), unit(0, 1)
    gallery = [{"person_id": 1, "embedding": a}, {"person_id": 2, "embedding": b}]
    assert find_best_face_match(unit(1, 0.05), gallery) == {
        "person_id": 1, "score": find_best_face_match(unit(1, 0.05), gallery)["score"], "ambiguous": False}
    assert find_best_face_match(unit(0, 0, 1), gallery) is None  # nobody above the threshold
    twin = unit(1, 0.01)
    both = [{"person_id": 1, "embedding": a}, {"person_id": 2, "embedding": twin}]
    assert find_best_face_match(a, both)["ambiguous"] is True
    one_person_two_photos = [{"person_id": 1, "embedding": a}, {"person_id": 1, "embedding": twin}]
    assert find_best_face_match(a, one_person_two_photos)["ambiguous"] is False  # same person is not a rival


def test_hybrid_face_unknown_falls_back_to_body():
    db, Session, lab, _ = seeded()
    no_face = add_person(db, "Asha", [lab.id], [("osnet", unit(1, 0))])
    with_face = add_person(db, "Bob", [lab.id], [("osnet", unit(0, 1)), ("arcface", unit(0, 0, 1))])
    db.commit()
    assert not person_has_face_enrolment(db, no_face.id) and person_has_face_enrolment(db, with_face.id)

    crop = np.zeros((200, 80, 3), dtype=np.uint8)
    face_unknown = {"status": "UNKNOWN", "person_id": None}

    def body(person):
        return {"status": "ALLOWED", "person_id": person.id, "color": "green", "message": "ok"}

    def run(person):
        with patch.object(stream_service, "SessionLocal", Session), \
             patch.object(stream_service, "identify_by_face", return_value=dict(face_unknown)), \
             patch.object(stream_service, "_identify_body", return_value=body(person)):
            return stream_service._run_identify_hybrid(crop, "LAB")

    assert run(no_face)["status"] == "ALLOWED"       # enrolled without a face: body decides
    assert run(with_face)["status"] == "ALLOWED"     # veto off (overhead cameras): body decides
    with patch.object(stream_service.settings, "FACE_VETO_BODY", True):
        assert run(with_face)["status"] == "VERIFY"  # veto on: face seen and rejected, body can never grant access

    with patch.object(stream_service, "SessionLocal", Session), \
         patch.object(stream_service, "identify_by_face", return_value=dict(face_unknown)), \
         patch.object(stream_service, "_identify_body", return_value=None):
        assert stream_service._run_identify_hybrid(crop, "LAB")["status"] == "UNKNOWN"


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("PASS", name)
