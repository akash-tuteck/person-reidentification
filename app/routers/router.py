from fastapi import APIRouter

from app.routers.endpoints import image_router, video_router, access_router
from app.routers.endpoints import stream_router
from app.routers.endpoints.viewer_router import viewer_router

api_router = APIRouter()
api_router.include_router(image_router.router, prefix="/process", tags=["Image Processing"])
api_router.include_router(video_router.router, prefix="/videos", tags=["Video Processing"])
api_router.include_router(access_router.router)
api_router.include_router(stream_router.router)
api_router.include_router(viewer_router)

