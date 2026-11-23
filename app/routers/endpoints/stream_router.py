"""
Phase 3 — Stream endpoints
POST  /api/v1/stream/start/{camera_id}   Start processing a camera
POST  /api/v1/stream/stop/{camera_id}    Stop a camera stream
GET   /api/v1/stream/status              List all running streams
GET   /api/v1/stream/status/{camera_id} Status of one camera
GET   /api/v1/stream/frame/{camera_id}  Latest annotated JPEG frame (single snapshot)
GET   /api/v1/stream/video/{camera_id}  Smooth MJPEG live video (open in browser)
GET   /api/v1/stream/events             Recent security events
"""

from datetime import datetime
from typing import List, Optional

import cv2
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import Response, StreamingResponse
from pydantic import BaseModel, ConfigDict
from sqlalchemy.orm import Session

from app.security import verify_api_key, verify_api_key_or_query
from app.database.session import get_db
from app.database.models import Camera, SecurityEvent
from app.services.stream_service import (
    start_stream,
    stop_stream,
    get_stream_status,
    get_all_stream_statuses,
    get_latest_frame,
    get_debug_info,
    mjpeg_frame_generator,
)


router = APIRouter(prefix="/api/v1/stream", tags=["Stream Processing"])

# ── Response models ───────────────────────────────────────────────────────────

class StreamStatusResponse(BaseModel):
    camera_id: int
    status: str


class SecurityEventResponse(BaseModel):
    id: str
    timestamp: datetime
    camera_id: int
    zone_id: int
    person_id: Optional[int] = None
    track_id: Optional[int] = None
    status: str
    similarity_score: Optional[float] = None
    snapshot_path: Optional[str] = None
    is_acknowledged: bool

    model_config = ConfigDict(from_attributes=True)


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.post("/start/{camera_id}", response_model=dict)
def start_camera_stream(
    camera_id: int,
    db: Session = Depends(get_db),
    _: str = Depends(verify_api_key),
):
    """
    Start processing video from a registered camera.
    The camera must exist in the database and be bound to a zone.
    The rtsp_url can be an RTSP address OR a local video file path (e.g. C:/videos/cctv.mp4).
    """
    camera = db.query(Camera).filter(Camera.id == camera_id).first()
    if not camera:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Camera ID {camera_id} not found.",
        )
    if not camera.zone_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Camera has no zone bound. Bind it to a zone first.",
        )

    result = start_stream(camera_id)
    return result


@router.post("/stop/{camera_id}", response_model=dict)
def stop_camera_stream(
    camera_id: int,
    _: str = Depends(verify_api_key),
):
    """Stop a running camera stream."""
    result = stop_stream(camera_id)
    return result


@router.get("/status", response_model=List[StreamStatusResponse])
def list_stream_statuses(_: str = Depends(verify_api_key)):
    """List all currently active/running streams."""
    return get_all_stream_statuses()


@router.get("/status/{camera_id}", response_model=StreamStatusResponse)
def camera_stream_status(
    camera_id: int,
    _: str = Depends(verify_api_key),
):
    """Get the status of a single camera stream."""
    return get_stream_status(camera_id)


@router.get("/debug/{camera_id}", response_model=dict)
def debug_stream(camera_id: int, _: str = Depends(verify_api_key_or_query)):
    """
    Debug endpoint — shows status, whether a frame has been stored, and any error.
    Use this if /stream/frame/{camera_id} returns 404 to find out why.
    """
    return get_debug_info(camera_id)


@router.get("/video/{camera_id}")
def stream_mjpeg_video(camera_id: int, _: str = Depends(verify_api_key_or_query)):
    """
    Returns a smooth MJPEG video stream for a running camera.
    Open in a browser as /api/v1/stream/video/{camera_id}?key=<API key>.
    """
    entry = get_debug_info(camera_id)
    if entry["status"] == "not_running" or entry["status"].startswith("ERROR"):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Camera {camera_id} stream is not running ({entry['status']}). Start it first.",
        )
    return StreamingResponse(
        mjpeg_frame_generator(camera_id),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


@router.get("/frame/{camera_id}", response_class=Response)
def get_camera_frame(camera_id: int, _: str = Depends(verify_api_key_or_query)):
    """
    Returns the latest annotated JPEG frame from a running camera stream.
    Open in a browser as /api/v1/stream/frame/{camera_id}?key=<API key>.
    """
    frame = get_latest_frame(camera_id)
    if frame is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No frame available for camera {camera_id}. Is the stream running?",
        )
    _, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
    return Response(content=jpeg.tobytes(), media_type="image/jpeg")


@router.get("/events", response_model=List[SecurityEventResponse])
def list_security_events(
    limit: int = 50,
    unacknowledged_only: bool = False,
    db: Session = Depends(get_db),
    _: str = Depends(verify_api_key),
):
    """
    List recent security events (violations, unknowns).
    Use unacknowledged_only=true to see only unreviewed alerts.
    """
    query = db.query(SecurityEvent)
    if unacknowledged_only:
        query = query.filter(SecurityEvent.is_acknowledged == False)
    events = (
        query.order_by(SecurityEvent.timestamp.desc())
        .limit(limit)
        .all()
    )
    return events


@router.post("/events/{event_id}/acknowledge", response_model=dict)
def acknowledge_event(
    event_id: str,
    db: Session = Depends(get_db),
    _: str = Depends(verify_api_key),
):
    """Mark a security event as acknowledged (reviewed by guard)."""
    event = db.query(SecurityEvent).filter(SecurityEvent.id == event_id).first()
    if not event:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Event {event_id} not found.",
        )
    event.is_acknowledged = True
    db.commit()
    return {"acknowledged": True, "event_id": event_id}
