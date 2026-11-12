
import cv2
from pathlib import Path


class ImageQualityService:

    @staticmethod
    def validate_crop(
        image_path,
        min_width=64,
        min_height=128,
        poor_threshold=20.0,
        good_threshold=50.0
    ):

        image_path = Path(image_path)

        # 1. Check whether image exists
        if not image_path.exists():
            return {
                "can_process": False,
                "quality_status": "invalid",
                "message": "Image file not found",
                "issues": ["file_not_found"]
            }

        # 2. Read image
        image = cv2.imread(str(image_path))

        if image is None:
            return {
                "can_process": False,
                "quality_status": "invalid",
                "message": "Unable to read image",
                "issues": ["invalid_image"]
            }

        # 3. Get image dimensions
        height, width = image.shape[:2]

        issues = []

        # 4. Validate dimensions
        if width < min_width:
            issues.append("width_too_small")

        if height < min_height:
            issues.append("height_too_small")

        # 5. Reject only if dimensions are insufficient
        if issues:
            return {
                "can_process": False,
                "quality_status": "invalid",
                "width": width,
                "height": height,
                "blur_score": None,
                "message": "Image dimensions are insufficient",
                "issues": issues
            }

        # 6. Convert image to grayscale
        gray = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2GRAY
        )

        # 7. Calculate blur score
        blur_score = cv2.Laplacian(
            gray,
            cv2.CV_64F
        ).var()

        blur_score = float(blur_score)

        # 8. Classify image quality
        if blur_score < poor_threshold:
            quality_status = "poor"
            message = "Poor image sharpness. Processing with warning."
            issues.append("image_too_blurry")

        elif blur_score < good_threshold:
            quality_status = "moderate"
            message = "Moderate image sharpness. Processing with warning."
            issues.append("moderate_sharpness")

        else:
            quality_status = "good"
            message = "Image quality is good."

        # 9. Return validation result
        return {
            "can_process": True,
            "quality_status": quality_status,
            "width": width,
            "height": height,
            "blur_score": round(blur_score, 2),
            "message": message,
            "issues": issues
        }
