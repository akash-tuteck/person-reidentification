"""
One-time script: extract ArcFace embeddings for all already-enrolled persons.
Run this ONCE after the upgrade so face recognition works without re-enrolling.

Usage:
  cd C:\person-reidentification
  venv\Scripts\python.exe extract_face_embeddings.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from pathlib import Path
import numpy as np

from app.database.session import SessionLocal, init_db
from app.database.models import Person, PersonPhoto, PersonEmbedding
from app.services.face_service import extract_face_embedding

def run():
    init_db()
    db = SessionLocal()

    persons = db.query(Person).filter(Person.is_active == True).all()
    print(f"\nFound {len(persons)} enrolled person(s).\n")

    total_extracted = 0

    for person in persons:
        print(f"── {person.name} (ID={person.id})")
        photos = db.query(PersonPhoto).filter(PersonPhoto.person_id == person.id).all()

        # Check which photos already have ArcFace embeddings
        existing_arcface_photo_ids = set(
            row.photo_id for row in
            db.query(PersonEmbedding)
               .filter(
                   PersonEmbedding.person_id == person.id,
                   PersonEmbedding.embedding_type == "arcface"
               ).all()
        )

        extracted = 0
        for photo in photos:
            photo_path = Path(photo.photo_path)

            # Skip if ArcFace already exists for this photo
            if photo.id in existing_arcface_photo_ids:
                print(f"   photo {photo.id} — ArcFace already exists, skipping")
                continue

            if not photo_path.exists():
                print(f"   photo {photo.id} — file not found: {photo_path}")
                continue

            face_emb = extract_face_embedding(photo_path)
            if face_emb is None:
                print(f"   photo {photo.id} — ❌ no face detected in {photo_path.name}")
                continue

            # Save ArcFace embedding to DB
            rec = PersonEmbedding(
                person_id=person.id,
                photo_id=photo.id,
                embedding=face_emb.tobytes(),
                model_name="arcface_buffalo_l",
                embedding_type="arcface",
            )
            db.add(rec)
            db.flush()
            extracted += 1
            total_extracted += 1
            print(f"   photo {photo.id} — ✅ ArcFace extracted from {photo_path.name}")

        if extracted == 0 and not existing_arcface_photo_ids:
            print(f"   ⚠️  No face found in ANY photo for {person.name}.")
            print(f"       → Please re-enroll with a clear front-facing portrait photo.")

        db.commit()
        print()

    db.close()
    print(f"Done. {total_extracted} new ArcFace embedding(s) saved.\n")
    if total_extracted == 0:
        print("⚠️  No new embeddings were extracted.")
        print("   This usually means the enrolled photos don't show a clear face.")
        print("   Solution: re-enroll each person using a clear ID/passport-style photo")
        print("   as photo_1, then add CCTV-angle shots as photo_2 to photo_5.\n")

if __name__ == "__main__":
    run()
