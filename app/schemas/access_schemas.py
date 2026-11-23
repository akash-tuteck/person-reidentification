import re
from datetime import datetime
from typing import List, Optional
from pydantic import BaseModel, ConfigDict, Field


def mask_rtsp_url(url: str) -> str:
    """
    Mask sensitive RTSP credentials in URLs.
    Example: rtsp://admin:secret123@192.168.1.50:554/stream -> rtsp://admin:***@192.168.1.50:554/stream
    """
    if not url:
        return url
    # Matches :password@ or //user:password@
    return re.sub(r'(://[^:]+):([^@]+)@', r'\1:***@', url)


# Zone schemas
class ZoneCreate(BaseModel):
    code: str = Field(..., description="Unique zone code (e.g. SERVER_ROOM, TRANSFORMER_YARD)")
    name: str = Field(..., description="Display name of the zone (e.g. Server Room)")
    description: Optional[str] = Field(None, description="Optional description of the zone")


class ZoneResponse(BaseModel):
    id: int
    code: str
    name: str
    description: Optional[str] = None
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)


# Camera schemas
class CameraCreate(BaseModel):
    name: str = Field(..., description="Camera identifier name (e.g. Server Room North Gate)")
    rtsp_url: str = Field(..., description="RTSP video stream URL")
    camera_type: str = Field("fixed", description="Camera type: fixed, ptz, thermal, anpr")
    zone_id: int = Field(..., description="Database ID of the zone this camera monitors")
    polygon_points: Optional[str] = Field(None, description="Optional JSON polygon coordinates for sub-zone")


class CameraResponse(BaseModel):
    id: int
    name: str
    rtsp_url: str
    camera_type: str
    zone_id: int
    zone_code: Optional[str] = None
    zone_name: Optional[str] = None
    polygon_points: Optional[str] = None
    is_active: bool
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)


# Person schemas
class PersonResponse(BaseModel):
    id: int
    name: str
    employee_code: Optional[str] = None
    department: Optional[str] = None
    is_active: bool
    photos_count: int = 0
    allowed_zones: List[str] = []
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)


class DeleteResponse(BaseModel):
    success: bool
    message: str
    person_id: int


# Identify schemas
class IdentifyResponse(BaseModel):
    status: str           # ALLOWED / UNAUTHORIZED / VERIFY / UNKNOWN / NO_DECISION / ERROR
    color: str            # green / red / amber / gray
    message: str          # human-readable result shown on screen / frame overlay
    zone_code: Optional[str] = None
    zone_name: Optional[str] = None
    person_id: Optional[int] = None
    person_name: Optional[str] = None
    employee_code: Optional[str] = None
    department: Optional[str] = None
    similarity_score: Optional[float] = None   # 0-100 scale
    quality_info: Optional[dict] = None
