from datetime import datetime
from sqlalchemy import (
    Column,
    Integer,
    String,
    Text,
    Boolean,
    Float,
    DateTime,
    ForeignKey,
    LargeBinary
)
from sqlalchemy.orm import relationship

from app.database.session import Base


class Zone(Base):
    __tablename__ = "zones"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    code = Column(String(50), unique=True, index=True, nullable=False)
    name = Column(String(100), nullable=False)
    description = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    cameras = relationship("Camera", back_populates="zone", cascade="all, delete-orphan")
    permissions = relationship("PersonZonePermission", back_populates="zone", cascade="all, delete-orphan")
    events = relationship("SecurityEvent", back_populates="zone")


class Camera(Base):
    __tablename__ = "cameras"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    name = Column(String(100), nullable=False)
    rtsp_url = Column(String(500), nullable=False)
    camera_type = Column(String(20), default="fixed", nullable=False)  # fixed, ptz, thermal, anpr
    zone_id = Column(Integer, ForeignKey("zones.id", ondelete="CASCADE"), nullable=False)
    polygon_points = Column(Text, nullable=True)  # JSON string of polygon coordinates for Method 2
    is_active = Column(Boolean, default=True, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    zone = relationship("Zone", back_populates="cameras")
    events = relationship("SecurityEvent", back_populates="camera")


class Person(Base):
    __tablename__ = "persons"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    name = Column(String(100), nullable=False)
    employee_code = Column(String(50), unique=True, index=True, nullable=True)
    department = Column(String(100), nullable=True)
    is_active = Column(Boolean, default=True, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    photos = relationship("PersonPhoto", back_populates="person", cascade="all, delete-orphan")
    embeddings = relationship("PersonEmbedding", back_populates="person", cascade="all, delete-orphan")
    permissions = relationship("PersonZonePermission", back_populates="person", cascade="all, delete-orphan")
    events = relationship("SecurityEvent", back_populates="person")


class PersonPhoto(Base):
    __tablename__ = "person_photos"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    person_id = Column(Integer, ForeignKey("persons.id", ondelete="CASCADE"), nullable=False)
    photo_path = Column(String(500), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    person = relationship("Person", back_populates="photos")
    embedding = relationship("PersonEmbedding", back_populates="photo", uselist=False, cascade="all, delete-orphan")


class PersonEmbedding(Base):
    __tablename__ = "person_embeddings"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    person_id = Column(Integer, ForeignKey("persons.id", ondelete="CASCADE"), nullable=False)
    photo_id = Column(Integer, ForeignKey("person_photos.id", ondelete="CASCADE"), nullable=True)
    embedding = Column(LargeBinary, nullable=False)   # 512-dim float32 binary (2048 bytes)
    model_name = Column(String(50), default="osnet_x1_0", nullable=False)
    # embedding_type: "osnet" = body Re-ID,  "arcface" = face biometrics (primary)
    embedding_type = Column(String(20), default="osnet", nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    person = relationship("Person", back_populates="embeddings")
    photo = relationship("PersonPhoto", back_populates="embedding")


class Role(Base):
    """Forward-compatible table for group-based permissions in future phases."""
    __tablename__ = "roles"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    name = Column(String(50), unique=True, nullable=False)
    description = Column(String(200), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    permissions = relationship("PersonZonePermission", back_populates="role")


class PersonZonePermission(Base):
    __tablename__ = "person_zone_permissions"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    person_id = Column(Integer, ForeignKey("persons.id", ondelete="CASCADE"), nullable=False)
    zone_id = Column(Integer, ForeignKey("zones.id", ondelete="CASCADE"), nullable=False)

    # Future-ready columns (nullable now, logic in later phases)
    role_id = Column(Integer, ForeignKey("roles.id", ondelete="SET NULL"), nullable=True)
    valid_from = Column(DateTime, nullable=True)
    valid_until = Column(DateTime, nullable=True)
    shift_start = Column(String(10), nullable=True)  # e.g., "09:00"
    shift_end = Column(String(10), nullable=True)    # e.g., "18:00"

    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    person = relationship("Person", back_populates="permissions")
    zone = relationship("Zone", back_populates="permissions")
    role = relationship("Role", back_populates="permissions")


class SecurityEvent(Base):
    """Records access decisions and security alerts."""
    __tablename__ = "security_events"

    id = Column(String(36), primary_key=True, index=True)  # UUID string
    timestamp = Column(DateTime, default=datetime.utcnow, index=True, nullable=False)
    camera_id = Column(Integer, ForeignKey("cameras.id"), nullable=False)
    zone_id = Column(Integer, ForeignKey("zones.id"), nullable=False)
    person_id = Column(Integer, ForeignKey("persons.id"), nullable=True)
    track_id = Column(Integer, nullable=True)
    status = Column(String(30), nullable=False)  # ALLOWED, UNAUTHORIZED, VERIFY, UNKNOWN, NO_DECISION
    similarity_score = Column(Float, nullable=True)
    snapshot_path = Column(String(500), nullable=True)
    clip_path = Column(String(500), nullable=True)
    is_acknowledged = Column(Boolean, default=False, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    camera = relationship("Camera", back_populates="events")
    zone = relationship("Zone", back_populates="events")
    person = relationship("Person", back_populates="events")
