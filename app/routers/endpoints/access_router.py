import shutil
import numpy as np
import torch
from pathlib import Path
from typing import List, Optional

from fastapi import (
    APIRouter,
    Depends,
    HTTPException,
    status,
    Form,
    File,
    UploadFile,
)
from sqlalchemy.orm import Session

from app.config import settings
from app.security import verify_api_key
from app.database.session import get_db
from app.database.models import (
    Zone,
    Camera,
    Person,
    PersonPhoto,
    PersonEmbedding,
    PersonZonePermission,
)
from app.schemas.access_schemas import (
    ZoneCreate,
    ZoneResponse,
    CameraCreate,
    CameraResponse,
    PersonResponse,
    DeleteResponse,
    IdentifyResponse,
    mask_rtsp_url,
)
from app.services.osnet_service import OSNetFeatureService
from app.services.image_quality_service import ImageQualityService
from app.services.identify_service import IdentifyService
from app.services.face_service import extract_face_embedding


router = APIRouter(prefix="/api/v1", tags=["Access Control"])


def _save_upload(upload: UploadFile, dest: Path) -> None:
    """Copy an upload to disk, refusing anything larger than MAX_FILE_SIZE."""
    written = 0
    with open(dest, "wb") as out:
        while chunk := upload.file.read(1024 * 1024):
            written += len(chunk)
            if written > settings.MAX_FILE_SIZE:
                raise HTTPException(
                    status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                    detail=f"Photo is larger than {settings.MAX_FILE_SIZE} bytes.",
                )
            out.write(chunk)


# ==========================================
# Zone Endpoints
# ==========================================

@router.post("/zones", response_model=ZoneResponse, status_code=status.HTTP_201_CREATED)
def create_zone(
    zone_in: ZoneCreate,
    db: Session = Depends(get_db),
    _: str = Depends(verify_api_key),
):
    """Register a new zone (e.g. SERVER_ROOM, TRANSFORMER_YARD, INVERTER_ROOM)."""
    code_normalized = zone_in.code.strip().upper()
    existing = db.query(Zone).filter(Zone.code == code_normalized).first()
    if existing:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Zone with code '{code_normalized}' already exists",
        )

    zone = Zone(
        code=code_normalized,
        name=zone_in.name.strip(),
        description=zone_in.description.strip() if zone_in.description else None,
    )
    db.add(zone)
    db.commit()
    db.refresh(zone)
    return zone


@router.get("/zones", response_model=List[ZoneResponse])
def list_zones(
    db: Session = Depends(get_db),
    _: str = Depends(verify_api_key),
):
    """List all registered access control zones."""
    return db.query(Zone).order_by(Zone.id.asc()).all()


# ==========================================
# Camera Endpoints
# ==========================================

@router.post("/cameras", response_model=CameraResponse, status_code=status.HTTP_201_CREATED)
def register_camera(
    camera_in: CameraCreate,
    db: Session = Depends(get_db),
    _: str = Depends(verify_api_key),
):
    """
    Register a camera and bind it to a zone.
    RTSP credentials are saved securely and never exposed in responses or logs.
    """
    zone = db.query(Zone).filter(Zone.id == camera_in.zone_id).first()
    if not zone:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Zone ID {camera_in.zone_id} not found. Create the zone first.",
        )

    camera = Camera(
        name=camera_in.name.strip(),
        rtsp_url=camera_in.rtsp_url.strip(),
        camera_type=camera_in.camera_type.lower().strip(),
        zone_id=zone.id,
        polygon_points=camera_in.polygon_points,
    )
    db.add(camera)
    db.commit()
    db.refresh(camera)

    return CameraResponse(
        id=camera.id,
        name=camera.name,
        rtsp_url=mask_rtsp_url(camera.rtsp_url),
        camera_type=camera.camera_type,
        zone_id=camera.zone_id,
        zone_code=zone.code,
        zone_name=zone.name,
        polygon_points=camera.polygon_points,
        is_active=camera.is_active,
        created_at=camera.created_at,
    )


@router.get("/cameras", response_model=List[CameraResponse])
def list_cameras(
    db: Session = Depends(get_db),
    _: str = Depends(verify_api_key),
):
    """List all registered cameras with their bound zone and masked RTSP URLs."""
    cameras = db.query(Camera).order_by(Camera.id.asc()).all()
    results = []
    for cam in cameras:
        results.append(
            CameraResponse(
                id=cam.id,
                name=cam.name,
                rtsp_url=mask_rtsp_url(cam.rtsp_url),
                camera_type=cam.camera_type,
                zone_id=cam.zone_id,
                zone_code=cam.zone.code if cam.zone else None,
                zone_name=cam.zone.name if cam.zone else None,
                polygon_points=cam.polygon_points,
                is_active=cam.is_active,
                created_at=cam.created_at,
            )
        )
    return results


# ==========================================
# Person Enrollment & Management Endpoints
# ==========================================

@router.post("/persons/enroll", response_model=PersonResponse, status_code=status.HTTP_201_CREATED)
def enroll_person(
    name: str = Form(..., description="Full name of person"),
    employee_code: Optional[str] = Form(None, description="Unique employee / contractor code"),
    department: Optional[str] = Form(None, description="Department name"),
    allowed_zone_codes: str = Form(..., description="Comma-separated zone codes (e.g. SERVER_ROOM,CONTROL_ROOM)"),
    photo_1: UploadFile = File(...,  description="Photo 1 (required) — front view"),
    photo_2: Optional[UploadFile] = File(None, description="Photo 2 (optional) — side view"),
    photo_3: Optional[UploadFile] = File(None, description="Photo 3 (optional) — back view"),
    photo_4: Optional[UploadFile] = File(None, description="Photo 4 (optional) — extra angle"),
    photo_5: Optional[UploadFile] = File(None, description="Photo 5 (optional) — extra angle"),
    db: Session = Depends(get_db),
    _: str = Depends(verify_api_key),
):
    """
    Enroll a person with 1 to 5 photos and assign their authorized zones.
    Validates crop quality, extracts 512-d OSNet embeddings + ArcFace face embeddings,
    and stores everything in the database and on disk.
    For best results: use a clear front-facing photo (passport/ID style) as photo_1.
    Additional CCTV-angle photos can be added as photo_2 to photo_5.
    """
    # Combine individual photo fields into one list (skip empty ones)
    photos = [p for p in [photo_1, photo_2, photo_3, photo_4, photo_5] if p is not None and p.filename]

    if len(photos) < 1:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="You must upload at least 1 photo for enrollment.",
        )

    # Check employee code uniqueness if provided
    clean_emp_code = employee_code.strip() if employee_code else None
    if clean_emp_code:
        existing_emp = db.query(Person).filter(Person.employee_code == clean_emp_code).first()
        if existing_emp:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Person with employee code '{clean_emp_code}' is already registered.",
            )

    # Validate and resolve allowed zones
    zone_codes = [c.strip().upper() for c in allowed_zone_codes.split(",") if c.strip()]
    if not zone_codes:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="At least one allowed zone code must be provided.",
        )

    zones = db.query(Zone).filter(Zone.code.in_(zone_codes)).all()
    found_codes = {z.code for z in zones}
    missing_codes = set(zone_codes) - found_codes
    if missing_codes:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Zone code(s) do not exist: {', '.join(sorted(missing_codes))}. Create them first.",
        )

    # Create Person in database
    person = Person(
        name=name.strip(),
        employee_code=clean_emp_code,
        department=department.strip() if department else None,
        is_active=True,
    )
    db.add(person)
    db.flush()  # Populates person.id

    person_dir = settings.GALLERY_DIR / str(person.id)
    person_dir.mkdir(parents=True, exist_ok=True)

    try:
        saved_photos_count = _enroll_photos(db, person, person_dir, photos)
        for zone in zones:
            db.add(PersonZonePermission(person_id=person.id, zone_id=zone.id))
        db.commit()
    except BaseException:
        # Nothing half-enrolled: no database rows and no photos left on disk.
        db.rollback()
        shutil.rmtree(person_dir, ignore_errors=True)
        raise
    db.refresh(person)

    return PersonResponse(
        id=person.id,
        name=person.name,
        employee_code=person.employee_code,
        department=person.department,
        is_active=person.is_active,
        photos_count=saved_photos_count,
        allowed_zones=sorted(found_codes),
        created_at=person.created_at,
    )


def _enroll_photos(db: Session, person: Person, person_dir: Path, photos: List[UploadFile]) -> int:
    """Save, validate and embed each photo (OSNet body + ArcFace face). Raises HTTPException on bad input."""
    extractor = OSNetFeatureService.get_extractor()

    saved_photos_count = 0
    for idx, upload in enumerate(photos, start=1):
        photo_filename = f"photo_{idx}.jpg"
        photo_path = person_dir / photo_filename

        _save_upload(upload, photo_path)

        # Validate image quality (dimensions & blur)
        quality = ImageQualityService.validate_crop(photo_path)
        if not quality["can_process"]:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Photo {idx} failed quality check: {quality['message']} (Issues: {quality.get('issues')}). Minimum 64x128 pixels required.",
            )

        # Extract OSNet 512-dim feature embedding
        features = extractor([str(photo_path)])
        if isinstance(features, torch.Tensor):
            features = features.detach().cpu().numpy()
        features = np.asarray(features)

        feature = features[0]
        norm = np.linalg.norm(feature)
        if not np.isfinite(norm) or norm == 0:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to generate valid embedding for photo {idx}.",
            )

        normalized_feature = (feature / norm).astype(np.float32)

        # Save embedding .npy alongside photo for quick local inspections
        npy_path = person_dir / f"embedding_{idx}.npy"
        np.save(npy_path, normalized_feature)

        # Create PersonPhoto record
        photo_record = PersonPhoto(
            person_id=person.id,
            photo_path=str(photo_path),
        )
        db.add(photo_record)
        db.flush()

        # Create PersonEmbedding record — OSNet body Re-ID (fallback)
        embedding_record = PersonEmbedding(
            person_id=person.id,
            photo_id=photo_record.id,
            embedding=normalized_feature.tobytes(),
            model_name="osnet_x1_0",
            embedding_type="osnet",
        )
        db.add(embedding_record)

        # Also try to extract an ArcFace face embedding (primary identifier)
        # This works best on photo_1 (clear front-facing portrait)
        face_emb = extract_face_embedding(photo_path)
        if face_emb is not None:
            face_record = PersonEmbedding(
                person_id=person.id,
                photo_id=photo_record.id,
                embedding=face_emb.tobytes(),
                model_name="arcface_buffalo_l",
                embedding_type="arcface",
            )
            db.add(face_record)
            print(f"[Enroll] ArcFace embedding extracted from photo {idx} for {person.name}")
        else:
            print(f"[Enroll] No face detected in photo {idx} — OSNet only for this photo")

        saved_photos_count += 1

    return saved_photos_count


@router.get("/persons", response_model=List[PersonResponse])
def list_persons(
    db: Session = Depends(get_db),
    _: str = Depends(verify_api_key),
):
    """List all enrolled persons, their enrolled photo counts, and allowed zones."""
    persons = db.query(Person).order_by(Person.id.asc()).all()
    results = []
    for p in persons:
        zone_codes = [perm.zone.code for perm in p.permissions if perm.zone]
        results.append(
            PersonResponse(
                id=p.id,
                name=p.name,
                employee_code=p.employee_code,
                department=p.department,
                is_active=p.is_active,
                photos_count=len(p.photos),
                allowed_zones=sorted(zone_codes),
                created_at=p.created_at,
            )
        )
    return results


@router.delete("/persons/{person_id}", response_model=DeleteResponse)
def delete_person(
    person_id: int,
    db: Session = Depends(get_db),
    _: str = Depends(verify_api_key),
):
    """
    Delete an enrolled person and cascade remove photos, embeddings, and permissions
    from both the database and the file system.
    """
    person = db.query(Person).filter(Person.id == person_id).first()
    if not person:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Person with ID {person_id} not found.",
        )

    # Remove disk directory
    person_dir = settings.GALLERY_DIR / str(person_id)
    if person_dir.exists():
        shutil.rmtree(person_dir, ignore_errors=True)

    # Delete from DB (cascades to photos, embeddings, and permissions)
    db.delete(person)
    db.commit()

    return DeleteResponse(
        success=True,
        message=f"Person ID {person_id} ({person.name}) deleted successfully.",
        person_id=person_id,
    )


# ==========================================
# Identify Endpoint (Phase 2)
# ==========================================

@router.post("/identify", response_model=IdentifyResponse, tags=["Access Control"])
def identify_person(
    zone_code: str = Form(..., description="Zone code to check access against (e.g. SERVER_ROOM)"),
    photo: UploadFile = File(..., description="Photo/crop of the person to identify"),
    db: Session = Depends(get_db),
    _: str = Depends(verify_api_key),
):
    """
    Identify a person from a photo and return their access decision for a zone.

    **5 possible results:**
    - **ALLOWED** (green) — known person, authorized for this zone
    - **UNAUTHORIZED** (red) — known person, NOT authorized for this zone
    - **VERIFY** (amber) — ambiguous match, manual verification required. Never grants access.
    - **UNKNOWN** (red) — no match found, possible intruder
    - **NO_DECISION** (gray) — image too small or blurry to process
    """
    import tempfile

    # Save uploaded photo to a temp file for processing
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        tmp_path = Path(tmp.name)

    try:
        _save_upload(photo, tmp_path)
        result = IdentifyService.identify(
            photo_path=tmp_path,
            zone_code=zone_code,
            db=db,
        )
    finally:
        tmp_path.unlink(missing_ok=True)

    return IdentifyResponse(**result)
