
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from app.config import settings
from app.routers import router


# Initialize FastAPI application
app = FastAPI(
    title=settings.APP_NAME,
    description="Image Processing and Person Re-Identification Backend",
    version="1.0.0",
    debug=settings.DEBUG
)


# Register API routers
app.include_router(router.api_router)


# Create static image directory
from pathlib import Path

UPLOAD_DIR = Path("app/uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


# Serve uploaded images (local development only)
app.mount(
    "/uploads",
    StaticFiles(directory=str(UPLOAD_DIR)),
    name="uploads"
)


# Health check API
@app.get("/")
def home():
    return {
        "success": True,
        "message": "Person Re-Identification API is running",
        "environment": settings.APP_ENV
    }


# Health status API
@app.get("/health")
def health_check():
    return {
        "success": True,
        "status": "healthy"
    }
