
import cv2
import numpy as np

from PIL import Image
from io import BytesIO
from pathlib import Path
from uuid import uuid4

from app.config import settings


BASE_DIR = Path(__file__).resolve().parent.parent
ORIGINAL_DIR = BASE_DIR / "uploads" / "original"
PROCESSED_DIR = BASE_DIR / "uploads" / "processed"

ORIGINAL_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


class ImageProcessingService:

    @staticmethod
    def process_image(file_bytes: bytes):

        if not file_bytes:
            raise ValueError("Uploaded image is empty")

        if len(file_bytes) > settings.MAX_FILE_SIZE:
            raise ValueError("Maximum image size is 10 MB")

        try:
            with Image.open(BytesIO(file_bytes)) as img:

                if img.format not in ["JPEG", "PNG", "WEBP"]:
                    raise ValueError("Only JPG, PNG and WEBP allowed")

                width, height = img.size

                if width * height > settings.MAX_IMAGE_PIXELS:
                    raise ValueError("Image resolution is too large")

                img.verify()
                image_format = img.format.lower()

        except ValueError:
            raise

        except Exception:
            raise ValueError("Invalid image file")

        image_array = np.frombuffer(file_bytes, dtype=np.uint8)

        image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

        if image is None:
            raise ValueError("Unable to decode image")

        original_height, original_width = image.shape[:2]

        scale = min(
            settings.MAX_IMAGE_DIMENSION / original_width,
            settings.MAX_IMAGE_DIMENSION / original_height,
            1.0
        )

        new_width = max(1, int(original_width * scale))
        new_height = max(1, int(original_height * scale))

        processed_image = cv2.resize(
            image,
            (new_width, new_height),
            interpolation=cv2.INTER_AREA
        )

        image_id = str(uuid4())

        extension = {
            "jpeg": "jpg",
            "png": "png",
            "webp": "webp"
        }[image_format]

        original_path = ORIGINAL_DIR / f"{image_id}.{extension}"

        original_path.write_bytes(file_bytes)

        processed_path = PROCESSED_DIR / f"{image_id}.jpg"

        success = cv2.imwrite(
            str(processed_path),
            processed_image,
            [cv2.IMWRITE_JPEG_QUALITY, 90]
        )

        if not success:
            original_path.unlink(missing_ok=True)
            raise ValueError("Unable to save processed image")

        return {
            "image_id": image_id,
            "original_width": original_width,
            "original_height": original_height,
            "processed_width": new_width,
            "processed_height": new_height,
            "original_path": str(original_path),
            "processed_path": str(processed_path)
        }
    
    @staticmethod
    def get_processed_image_path(image_id: str):
        image_path = PROCESSED_DIR / f"{image_id}.jpg"

        if not image_path.exists():
            return None
            
        return image_path
