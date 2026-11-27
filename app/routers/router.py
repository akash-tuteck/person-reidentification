from fastapi import APIRouter

from app.routers.endpoints import (
    employee_router,
    image_router,
    video_router,
)

api_router = APIRouter()
api_router.include_router(employee_router.router, prefix="/employees", tags=["Employee Registry"])
api_router.include_router(image_router.router, prefix="/process", tags=["Image Processing"])
api_router.include_router(video_router.router, prefix="/videos", tags=["Video Processing"])
