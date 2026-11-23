"""
Phase 3 — Stream Processing Service  (v3 — Hybrid Face + Body Pipeline)

Architecture:
  - Dedicated capture thread  → deque(maxlen=1) always holds the LATEST frame
  - AI thread                 → pulls latest frame, runs YOLO + ByteTrack every frame
  - For UNVERIFIED tracks:
      1. Face crop (upper 40%) → ArcFace (PRIMARY — works on clear portrait photos)
      2. If no face visible   → OSNet body Re-ID (FALLBACK — clothing-based)
  - CONFIRMED tracks          → skip all AI, just redraw label (zero CPU cost)
  - CROSSOVER tracks          → hold last result, re-verify after IoU < 0.3

Key improvements in v3:
  ✅ ArcFace matches a clear enrolled photo to CCTV footage (OSNet cannot)
  ✅ Capture thread prevents frame lag / dropped frames
  ✅ Aspect ratio gate removed — close-up persons use face crop instead
  ✅ state machine unchanged: UNVERIFIED → CONFIRMED → CROSSOVER
"""

import os
import time
import uuid
import threading
import traceback
import tempfile
import numpy as np
import cv2

from collections import deque, defaultdict
from datetime import datetime
from pathlib import Path
from sqlalchemy.orm import Session

from app.database.session import SessionLocal
from app.database.models import Camera, SecurityEvent
from app.schemas.access_schemas import mask_rtsp_url
from app.services.identify_service import (
    IdentifyService,
    identify_by_face,
    person_has_face_enrolment,
)
from app.config import settings


# ── Constants ─────────────────────────────────────────────────────────────────
YOLO_TRACKER         = "bytetrack.yaml"

FRAMES_TO_CONFIRM    = 2     # consecutive good frames needed to CONFIRM a track
FRAME_SKIP           = 4     # re-identify an UNVERIFIED track at most every Nth frame
MAX_IDENTIFY_PER_FRAME = 2   # identification is slow; a crowd must never stall detection/drawing
TRACK_CLEANUP_AFTER  = 60    # frames without sighting → purge state

CROSSOVER_IOU_ENTER  = 0.50
CROSSOVER_IOU_EXIT   = 0.30

DWELL_SECONDS        = 5.0
COOLDOWN_SECONDS     = 120.0

# Quality gate — only size/blur, NO aspect ratio (close-up persons are still valid)
MIN_BODY_HEIGHT      = 48    # px — reduced from 64, CCTV crops can be small
MIN_FACE_WIDTH       = 30    # px — reduced from 40, CCTV faces are often small
MIN_SHARPNESS        = 30.0  # Laplacian variance — reduced from 60, CCTV is often slightly blurry

MJPEG_FPS = 25

# Track states
STATE_UNVERIFIED = "UNVERIFIED"
STATE_CONFIRMED  = "CONFIRMED"
STATE_CROSSOVER  = "CROSSOVER"

STATUS_COLORS = {
    "ALLOWED":      (0, 200, 0),
    "UNAUTHORIZED": (0, 0, 220),
    "UNKNOWN":      (0, 0, 180),
    "VERIFY":       (0, 165, 255),
    "NO_DECISION":  (120, 120, 120),
    "PROCESSING":   (120, 120, 120),
}


# ── Global state ──────────────────────────────────────────────────────────────
_active_streams: dict = {}
_lock = threading.Lock()


# One detector shared by all cameras (predict() only; each stream owns its tracker, see _new_tracker).
_yolo_model = None
_yolo_lock = threading.Lock()


def _get_yolo():
    global _yolo_model
    with _yolo_lock:
        if _yolo_model is None:
            from ultralytics import YOLO
            _yolo_model = YOLO(settings.YOLO_MODEL)
    return _yolo_model


def _new_tracker():
    """A tracker private to one stream, so track ids never mix between cameras."""
    from ultralytics.trackers.byte_tracker import BYTETracker
    from ultralytics.utils import YAML, IterableSimpleNamespace
    from ultralytics.utils.checks import check_yaml
    # track_buffer (30 in bytetrack.yaml) counts processed frames, so a lost person is kept ~10 s at CPU speeds
    cfg = YAML.load(check_yaml(YOLO_TRACKER))
    cfg["match_thresh"] = settings.TRACK_MATCH_THRESH
    return BYTETracker(IterableSimpleNamespace(**cfg))


def _clamp_box(xyxy, width, height):
    """Integer box inside the frame, or None if it is too small to use."""
    x1, y1, x2, y2 = (int(v) for v in xyxy)
    x1, y1, x2, y2 = max(0, x1), max(0, y1), min(width, x2), min(height, y2)
    return (x1, y1, x2, y2) if x2 > x1 + 10 and y2 > y1 + 10 else None


# ── Helpers ───────────────────────────────────────────────────────────────────

def _iou(a, b) -> float:
    xi = max(a[0], b[0]); yi = max(a[1], b[1])
    xa = min(a[2], b[2]); ya = min(a[3], b[3])
    inter = max(0, xa - xi) * max(0, ya - yi)
    union = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
    return inter / union if union > 0 else 0.0


def _is_blurry(crop) -> bool:
    """True if the crop is too blurry for reliable identification."""
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var() < MIN_SHARPNESS


def _annotate(frame, x1, y1, x2, y2, status: str, label: str):
    color = STATUS_COLORS.get(status, (120, 120, 120))
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
    cv2.rectangle(frame, (x1, max(0, y1 - th - 8)), (x1 + tw + 4, y1), color, -1)
    cv2.putText(frame, label, (x1 + 2, max(th, y1 - 4)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)


def _save_evidence(frame, camera_id, zone_code, status) -> str:
    ts    = datetime.utcnow().strftime("%Y%m%d_%H%M%S_%f")
    fname = f"cam{camera_id}_{zone_code}_{status}_{ts}.jpg"
    path  = settings.EVIDENCE_DIR / fname
    cv2.imwrite(str(path), frame)
    return str(path)


def _save_event(camera_id, zone_id, person_id, track_id, status, score, snapshot_path):
    db: Session = SessionLocal()
    try:
        db.add(SecurityEvent(
            id=str(uuid.uuid4()), timestamp=datetime.utcnow(),
            camera_id=camera_id, zone_id=zone_id,
            person_id=person_id, track_id=int(track_id) if track_id else None,
            status=status, similarity_score=score,
            snapshot_path=snapshot_path, is_acknowledged=False,
        ))
        db.commit()
    finally:
        db.close()


def _draw_track(frame, tid, x1, y1, x2, y2, result, dwell_start, last_alert,
                last_labels, camera_id, zone_id, zone_code, now, confirmed):
    """Build label, check dwell/cooldown, fire alert, draw box."""
    final_status = result["status"]
    is_violation = final_status in ("UNAUTHORIZED", "UNKNOWN")

    if tid not in dwell_start:
        dwell_start[tid] = now
    dwell = now - dwell_start[tid]

    name = result.get("person_name") or "UNKNOWN"
    if   final_status == "ALLOWED":       label = f"{name} | ALLOWED"
    elif final_status == "UNAUTHORIZED":  label = f"{name} | NOT AUTHORIZED: {zone_code}"
    elif final_status == "UNKNOWN":       label = f"UNKNOWN PERSON | {zone_code}"
    elif final_status == "VERIFY":        label = f"{name}? | VERIFY"
    else:                                  label = "PROCESSING..."

    if is_violation:
        label += f" [{dwell:.0f}s]"

    # Tag whether result came from face recognition or body Re-ID
    method = result.get("method", "")
    if method:
        label += f" [{method}]"

    last_labels[tid] = (final_status, label)
    _annotate(frame, x1, y1, x2, y2, final_status, label)

    # Save after drawing so the evidence shows the box of the person who triggered it.
    # Alert only on a CONFIRMED identity: one bad frame must not raise an intruder alert.
    if confirmed and is_violation and dwell >= DWELL_SECONDS and now - last_alert.get(tid, 0) >= COOLDOWN_SECONDS:
        snap = _save_evidence(frame, camera_id, zone_code, final_status)
        _save_event(camera_id, zone_id, result.get("person_id"), tid,
                    final_status, result.get("similarity_score"), snap)
        last_alert[tid] = now


def _identify_body(crop_bgr, zone_code: str) -> dict | None:
    """OSNet body Re-ID. None when the crop is too small or blurry to trust."""
    if crop_bgr.shape[0] < MIN_BODY_HEIGHT or _is_blurry(crop_bgr):
        return None

    fd, tmp_name = tempfile.mkstemp(suffix=".jpg")
    os.close(fd)
    tmp_path = Path(tmp_name)
    try:
        cv2.imwrite(tmp_name, crop_bgr)
        db = SessionLocal()
        try:
            return IdentifyService.identify(photo_path=tmp_path, zone_code=zone_code, db=db)
        finally:
            db.close()
    finally:
        tmp_path.unlink(missing_ok=True)


def _run_identify_hybrid(crop_bgr, zone_code: str) -> dict | None:
    """
    Face first (ArcFace), body (OSNet) as fallback.

    A face that matched nobody does not stop the body check: the person may have been enrolled
    without a usable face photo. But a body match must not override a face that was seen and
    rejected for someone who HAS a face enrolled; that case is downgraded to VERIFY.
    """
    db = SessionLocal()
    try:
        face_result = identify_by_face(crop_bgr, zone_code, db)
        if face_result is not None and face_result["status"] != "UNKNOWN":
            face_result["method"] = "face"
            return face_result

        body_result = _identify_body(crop_bgr, zone_code)
        if body_result is not None and body_result["status"] in ("ALLOWED", "UNAUTHORIZED"):
            if (settings.FACE_VETO_BODY and face_result is not None
                    and person_has_face_enrolment(db, body_result["person_id"])):
                body_result.update(
                    status="VERIFY", color="amber",
                    message="Body matches but the face does not. Manual verification required.")
            body_result["method"] = "body"
            return body_result
    finally:
        db.close()

    # face seen but unmatched -> UNKNOWN; no face and no usable body -> nothing yet
    if face_result is not None:
        face_result["method"] = "face"
        return face_result
    if body_result is not None:
        body_result["method"] = "body"
    return body_result


# ── Dedicated capture thread ──────────────────────────────────────────────────

def _capture_thread(source: str, frame_buf: deque, stop_event: threading.Event,
                    status_dict: dict):
    """
    Continuously reads frames from the video source into a 1-slot deque.
    For video files: paced to native FPS so playback speed is correct.
    For RTSP: frames arrive naturally at camera FPS — no pacing needed.
    The AI thread always pulls the LATEST frame, never an old buffered one.
    """
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        status_dict["capture"] = f"ERROR: cannot open {mask_rtsp_url(source)}"
        return

    is_rtsp = str(source).startswith("rtsp://")

    # For video files, pace reads to match the native FPS (e.g. 25 fps = 40ms/frame)
    native_fps   = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frame_delay  = 1.0 / native_fps   # seconds between frames

    while not stop_event.is_set():
        t0 = time.time()

        ret, frame = cap.read()
        if ret:
            frame_buf.append(frame)   # deque(maxlen=1) keeps only the latest
            status_dict["seq"] += 1   # lets the AI loop tell a new frame from the one it already did
        else:
            if is_rtsp:
                # RTSP disconnect — wait briefly and reconnect
                cap.release()
                time.sleep(1.0)
                cap = cv2.VideoCapture(source)
                native_fps  = cap.get(cv2.CAP_PROP_FPS) or 25.0
                frame_delay = 1.0 / native_fps
            else:
                # Video file ended — loop back to start
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                status_dict["loop"] = True

        # Pace to native FPS for video files so the video plays at normal speed
        if not is_rtsp:
            elapsed = time.time() - t0
            sleep_t = frame_delay - elapsed
            if sleep_t > 0:
                time.sleep(sleep_t)

    cap.release()


# ── Main stream worker ────────────────────────────────────────────────────────

def _stream_worker(camera_id: int, stop_event: threading.Event):
    try:
        _stream_worker_inner(camera_id, stop_event)
    except Exception as e:
        err = traceback.format_exc()
        print(f"[STREAM ERROR camera {camera_id}]:\n{err}")
        with _lock:
            if camera_id in _active_streams:
                _active_streams[camera_id]["status"] = f"ERROR: {e}"
                _active_streams[camera_id]["error"]  = err
    finally:
        stop_event.set()   # also ends the capture thread when the worker dies on its own


def _stream_worker_inner(camera_id: int, stop_event: threading.Event):

    # ── Load camera info ──────────────────────────────────────────────────────
    db: Session = SessionLocal()
    try:
        camera = db.query(Camera).filter(Camera.id == camera_id).first()
        if not camera:
            with _lock:
                if camera_id in _active_streams:
                    _active_streams[camera_id]["status"] = "ERROR: camera not found"
            return
        source    = camera.rtsp_url
        zone_code = camera.zone.code if camera.zone else None
        zone_id   = camera.zone_id
    finally:
        db.close()

    if not zone_code:
        with _lock:
            if camera_id in _active_streams:
                _active_streams[camera_id]["status"] = "ERROR: no zone bound"
        return

    # ── Start dedicated capture thread ────────────────────────────────────────
    frame_buf   = deque(maxlen=1)       # always holds the latest frame only
    cap_status  = {"capture": "ok", "loop": False, "seq": 0}
    cap_thread  = threading.Thread(
        target=_capture_thread,
        args=(source, frame_buf, stop_event, cap_status),
        daemon=True,
        name=f"capture-cam-{camera_id}",
    )
    cap_thread.start()

    # Wait until the first frame arrives (max 10 seconds)
    waited = 0.0
    while not frame_buf and waited < 10.0 and not stop_event.is_set():
        time.sleep(0.1); waited += 0.1

    if not frame_buf:
        with _lock:
            if camera_id in _active_streams:
                _active_streams[camera_id]["status"] = cap_status.get("capture", "ERROR: no frame")
        return

    print(f"[STREAM] Camera {camera_id} capture started: {mask_rtsp_url(source)}")

    # ── Load YOLO ─────────────────────────────────────────────────────────────
    with _lock:
        if camera_id in _active_streams:
            _active_streams[camera_id]["status"] = "loading_yolo"

    yolo = _get_yolo()
    tracker = _new_tracker()

    with _lock:
        if camera_id in _active_streams:
            _active_streams[camera_id]["status"] = "running"

    # ── Per-track state ───────────────────────────────────────────────────────
    track_state:      dict = {}   # tid -> STATE_*
    track_result:     dict = {}   # tid -> confirmed result dict
    good_frame_count: dict = {}   # tid -> consecutive good identify frames
    last_seen:        dict = {}   # tid -> frame_idx when last detected
    dwell_start:      dict = {}
    last_alert:       dict = {}
    last_labels:      dict = {}   # tid -> (status, label) for redraw on skipped frames
    last_identified:  dict = {}   # tid -> frame_idx of its last identification attempt

    frame_idx = 0
    last_seq  = 0

    while not stop_event.is_set():
        # ── Video file loop reset ─────────────────────────────────────────────
        if cap_status.get("loop"):
            cap_status["loop"] = False
            track_state.clear(); track_result.clear(); good_frame_count.clear()
            last_seen.clear(); dwell_start.clear(); last_labels.clear(); last_alert.clear()
            last_identified.clear(); tracker = _new_tracker()
            frame_idx = 0

        # ── Get latest frame from buffer ──────────────────────────────────────
        # Never analyse the same frame twice: it would feed ByteTrack zero motion and make
        # "N consecutive frames" mean one image seen N times.
        seq = cap_status["seq"]
        if not frame_buf or seq == last_seq:
            time.sleep(0.005)
            continue
        last_seq = seq
        frame = frame_buf[-1]   # always the most recent frame

        frame_idx += 1

        # ── Store raw frame IMMEDIATELY so the viewer always has something ────
        # This prevents the blank screen while YOLO is running
        with _lock:
            if camera_id in _active_streams and _active_streams[camera_id]["frame"] is None:
                _active_streams[camera_id]["frame"] = frame

        # ── Detect every person, then track. Detections the tracker has not confirmed yet are still
        # drawn ("pending"): ByteTrack only reports a new person after two matched frames, which at
        # CPU speeds hid 15-20% of the people the detector found.
        with _yolo_lock:
            results = yolo.predict(
                source=frame,
                classes=[0],
                conf=settings.YOLO_CONF,
                imgsz=settings.YOLO_IMGSZ,
                verbose=False,
            )
        detections = results[0].boxes.cpu().numpy()
        height, width = frame.shape[:2]

        tracked, matched_idx = [], set()
        for x1, y1, x2, y2, tid, _score, _cls, idx in tracker.update(detections, frame):
            box = _clamp_box((x1, y1, x2, y2), width, height)
            matched_idx.add(int(idx))
            if box:
                tracked.append((int(tid), *box))

        pending = [box for i in range(len(detections)) if i not in matched_idx
                   for box in [_clamp_box(detections.xyxy[i], width, height)] if box]

        now       = time.time()
        boxes_map = {tid: (x1, y1, x2, y2) for tid, x1, y1, x2, y2 in tracked}

        # ── Update last_seen ──────────────────────────────────────────────────
        for tid in boxes_map:
            last_seen[tid] = frame_idx

        # ── Cleanup stale tracks ──────────────────────────────────────────────
        stale = [tid for tid, lf in list(last_seen.items())
                 if frame_idx - lf > TRACK_CLEANUP_AFTER]
        for tid in stale:
            for d in (last_seen, track_state, track_result, good_frame_count,
                      dwell_start, last_alert, last_labels, last_identified):
                d.pop(tid, None)

        # ── Crossover detection ───────────────────────────────────────────────
        tid_list = list(boxes_map.keys())
        for i in range(len(tid_list)):
            for j in range(i + 1, len(tid_list)):
                ta, tb = tid_list[i], tid_list[j]
                if _iou(boxes_map[ta], boxes_map[tb]) >= CROSSOVER_IOU_ENTER:
                    if track_state.get(ta) == STATE_CONFIRMED:
                        track_state[ta] = STATE_CROSSOVER
                    if track_state.get(tb) == STATE_CONFIRMED:
                        track_state[tb] = STATE_CROSSOVER

        # ── CROSSOVER separation check (IoU < 0.3) ───────────────────────────
        for tid in list(boxes_map.keys()):
            if track_state.get(tid) != STATE_CROSSOVER:
                continue
            my_bbox = boxes_map[tid]
            max_iou = max(
                (_iou(my_bbox, boxes_map[other]) for other in boxes_map if other != tid),
                default=0.0,
            )
            if max_iou < CROSSOVER_IOU_EXIT:
                # Separated — reset for re-verification
                track_state[tid]      = STATE_UNVERIFIED
                good_frame_count[tid] = 0

        # ── Pick who to identify this frame: the tracks waiting longest, at most N ──
        never = frame_idx - FRAME_SKIP
        due = sorted(
            (tid for tid, *_ in tracked
             if track_state.get(tid, STATE_UNVERIFIED) == STATE_UNVERIFIED
             and frame_idx - last_identified.get(tid, never) >= FRAME_SKIP),
            key=lambda tid: last_identified.get(tid, never),
        )[:MAX_IDENTIFY_PER_FRAME]

        # ── Draw frame ────────────────────────────────────────────────────────
        annotated = frame.copy()

        for (tid, x1, y1, x2, y2) in tracked:
            state = track_state.get(tid, STATE_UNVERIFIED)

            # CONFIRMED → skip all AI, just redraw ────────────────────────────
            if state == STATE_CONFIRMED:
                _draw_track(annotated, tid, x1, y1, x2, y2,
                            track_result[tid], dwell_start, last_alert,
                            last_labels, camera_id, zone_id, zone_code, now, confirmed=True)
                continue

            # CROSSOVER → tracks overlapping, hold last result ─────────────────
            if state == STATE_CROSSOVER:
                result = track_result.get(tid)
                if result:
                    name  = result.get("person_name") or "UNKNOWN"
                    label = f"[CROSSING] {name}"
                    last_labels[tid] = ("VERIFY", label)
                    _annotate(annotated, x1, y1, x2, y2, "VERIFY", label)
                else:
                    _annotate(annotated, x1, y1, x2, y2, "NO_DECISION", "CROSSING...")
                continue

            # UNVERIFIED → identify only when it is this track's turn ──────────
            result = None
            if tid in due:
                last_identified[tid] = frame_idx
                result = _run_identify_hybrid(frame[y1:y2, x1:x2], zone_code)
            if result is None:
                if tid in last_labels:
                    s, lbl = last_labels[tid]
                    _annotate(annotated, x1, y1, x2, y2, s, lbl)
                else:
                    _annotate(annotated, x1, y1, x2, y2, "NO_DECISION", "PROCESSING...")
                continue

            status = result["status"]

            # Count consecutive frames with same status
            if status not in ("NO_DECISION", "ERROR"):
                prev = track_result.get(tid)
                if prev and prev["status"] == status:
                    good_frame_count[tid] = good_frame_count.get(tid, 0) + 1
                else:
                    good_frame_count[tid] = 1
                track_result[tid] = result

            # Promote to CONFIRMED once stable
            if good_frame_count.get(tid, 0) >= FRAMES_TO_CONFIRM:
                track_state[tid] = STATE_CONFIRMED
                method = result.get("method", "?")
                print(f"[STREAM] Cam {camera_id} track {tid} CONFIRMED "
                      f"→ {result.get('person_name') or 'UNKNOWN'} ({status}) via {method}")

            _draw_track(annotated, tid, x1, y1, x2, y2,
                        result, dwell_start, last_alert,
                        last_labels, camera_id, zone_id, zone_code, now,
                        confirmed=track_state.get(tid) == STATE_CONFIRMED)

        for (x1, y1, x2, y2) in pending:
            _annotate(annotated, x1, y1, x2, y2, "NO_DECISION", "DETECTING...")

        cv2.putText(annotated, f"People: {len(tracked) + len(pending)}", (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        # Store frame for viewer
        with _lock:
            if camera_id in _active_streams:
                _active_streams[camera_id]["frame"] = annotated

    with _lock:
        if camera_id in _active_streams:
            _active_streams[camera_id]["status"] = "stopped"


# ── MJPEG generator ───────────────────────────────────────────────────────────

def mjpeg_frame_generator(camera_id: int):
    """Yields MJPEG chunks; ends when the stream is stopped. Encodes each new frame once per viewer."""
    interval = 1.0 / MJPEG_FPS
    last_sent = None
    while True:
        t0 = time.time()
        with _lock:
            entry = _active_streams.get(camera_id)
            if entry is None or entry["status"].startswith("ERROR"):
                return
            frame = entry.get("frame")
        if frame is not None and frame is not last_sent:
            ok, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
            if ok:
                last_sent = frame
                yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
                       + jpeg.tobytes() + b"\r\n")
        sleep_t = interval - (time.time() - t0)
        if sleep_t > 0:
            time.sleep(sleep_t)


# ── Public API ─────────────────────────────────────────────────────────────────

def start_stream(camera_id: int) -> dict:
    stop_event = threading.Event()
    t = threading.Thread(target=_stream_worker, args=(camera_id, stop_event),
                         daemon=True, name=f"stream-cam-{camera_id}")
    with _lock:
        current = _active_streams.get(camera_id)
        if current and current["status"] in ("starting", "loading_yolo", "running"):
            return {"started": False, "message": f"Camera {camera_id} is already running"}
        _active_streams[camera_id] = {
            "thread": t, "stop_event": stop_event,
            "status": "starting", "frame": None, "error": None,
        }
    t.start()
    return {"started": True, "message": f"Camera {camera_id} stream started"}


def stop_stream(camera_id: int) -> dict:
    with _lock:
        entry = _active_streams.get(camera_id)
    if not entry:
        return {"stopped": False, "message": f"Camera {camera_id} is not running"}
    entry["stop_event"].set()
    entry["thread"].join(timeout=5)
    with _lock:
        _active_streams.pop(camera_id, None)
    return {"stopped": True, "message": f"Camera {camera_id} stream stopped"}


def get_stream_status(camera_id: int) -> dict:
    with _lock:
        entry = _active_streams.get(camera_id)
    if not entry:
        return {"camera_id": camera_id, "status": "not_running"}
    return {"camera_id": camera_id, "status": entry["status"]}


def get_all_stream_statuses() -> list:
    with _lock:
        return [{"camera_id": cid, "status": e["status"]}
                for cid, e in _active_streams.items()]


def get_latest_frame(camera_id: int):
    with _lock:
        entry = _active_streams.get(camera_id)
        return entry.get("frame") if entry else None


def get_debug_info(camera_id: int) -> dict:
    with _lock:
        entry = _active_streams.get(camera_id)
    if not entry:
        return {"camera_id": camera_id, "status": "not_running",
                "has_frame": False, "error": None}
    return {
        "camera_id": camera_id,
        "status":    entry["status"],
        "has_frame": entry.get("frame") is not None,
        "error":     entry.get("error"),
    }
