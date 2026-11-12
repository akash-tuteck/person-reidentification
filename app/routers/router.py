from fastapi import APIRouter

from app.routers.endpoints import image_router, video_router

api_router = APIRouter()
# api_router.include_router(health.router, prefix="/health", tags=["Health"])
# api_router.include_router(images.router, prefix="/images", tags=["Images"])
# # api_router.include_router(processing.router, prefix="/processing", tags=["Image Processing"])
# api_router.include_router(history.router, prefix="/history", tags=["History"])
api_router.include_router(image_router.router, prefix="/process", tags=["Image Processing"])
api_router.include_router(video_router.router, prefix="/videos", tags=["Video Processing"])
