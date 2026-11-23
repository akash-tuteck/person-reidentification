import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()


def _required(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"{name} must be set (see .env.example)")
    return value


class Settings:
    APP_NAME = os.getenv("APP_NAME", "Person Re-Identification & Access Control API")
    APP_ENV = os.getenv("APP_ENV", "development")
    DEBUG = os.getenv("DEBUG", "False").lower() == "true"

    # Auth is OFF for now (no auth middleware yet). Set AUTH_ENABLED=true to require X-API-Key again.
    AUTH_ENABLED = os.getenv("AUTH_ENABLED", "false").lower() == "true"
    API_KEY = _required("API_KEY") if AUTH_ENABLED else os.getenv("API_KEY", "")

    # SQLite for a single edge box; use postgresql+psycopg2://user:pass@host/db when shared.
    DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./data/production.db")

    # Upload and image limits
    MAX_FILE_SIZE = int(os.getenv("MAX_FILE_SIZE", 10485760))
    MAX_IMAGE_DIMENSION = int(os.getenv("MAX_IMAGE_DIMENSION", 1024))
    MAX_IMAGE_PIXELS = int(os.getenv("MAX_IMAGE_PIXELS", 25000000))

    # Person detector. yolov8n misses people in real CCTV (top-down angle, small, partly hidden);
    # yolov8m found 5x more on the sample clips. Use yolov8s on very weak CPUs.
    YOLO_MODEL = os.getenv("YOLO_MODEL", "yolov8m.pt")
    YOLO_CONF = float(os.getenv("YOLO_CONF", 0.25))
    YOLO_IMGSZ = int(os.getenv("YOLO_IMGSZ", 640))
    # Tracker association threshold. CPU runs analyse only a few frames/s, so people move far between
    # frames; 0.95 (vs ultralytics 0.8) tracked 88% instead of 61% of detections on the CCTV clip.
    # Body (OSNet) similarity needed to call someone known. Measured on overhead CCTV: strangers scored up
    # to 0.58, the enrolled man 0.73+, so 0.65 keeps a safe margin. Do not go below ~0.62 without more tests.
    BODY_MATCH_THRESHOLD = float(os.getenv("BODY_MATCH_THRESHOLD", 0.65))
    # When a face is seen and does NOT match someone who has a face enrolled, downgrade a body match to
    # VERIFY. Right for door/face cameras; wrong for overhead cameras where faces never match.
    FACE_VETO_BODY = os.getenv("FACE_VETO_BODY", "false").lower() == "true"
    TRACK_MATCH_THRESH = float(os.getenv("TRACK_MATCH_THRESH", 0.95))

    # Gallery and evidence storage paths
    GALLERY_DIR = Path(os.getenv("GALLERY_DIR", "./data/gallery"))
    EVIDENCE_DIR = Path(os.getenv("EVIDENCE_DIR", "./data/evidence"))


settings = Settings()

settings.GALLERY_DIR.mkdir(parents=True, exist_ok=True)
settings.EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)
Path("./data").mkdir(parents=True, exist_ok=True)
