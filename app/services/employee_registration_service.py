
import json
import re
import shutil
import uuid

from pathlib import Path

import cv2
import numpy as np
import torch

from app.config import settings
from app.services.osnet_service import OSNetFeatureService
from app.services.image_quality_service import ImageQualityService


BASE_DIR = Path(__file__).resolve().parent.parent

EMPLOYEE_REGISTRY_DIR = (
    BASE_DIR / "uploads" / "employee_registry"
)


class EmployeeRegistrationService:

    MAX_REFERENCE_PHOTOS = 5

    @classmethod
    def register_employee(
        cls,
        employee_id: str,
        employee_name: str,
        files: list
    ) -> dict:

        # 1. Validate employee information
        employee_id = employee_id.strip()
        employee_name = employee_name.strip()

        if not re.fullmatch(r"[A-Za-z0-9_-]{1,50}", employee_id):
            raise ValueError(
                "Employee ID must contain only letters, numbers, "
                "hyphens or underscores (maximum 50 characters)."
            )

        if not employee_name or len(employee_name) > 100:
            raise ValueError("Invalid employee name.")

        if not files:
            raise ValueError("Upload at least one reference photo.")

        if len(files) > cls.MAX_REFERENCE_PHOTOS:
            raise ValueError(
                f"Maximum {cls.MAX_REFERENCE_PHOTOS} reference photos allowed."
            )

        EMPLOYEE_REGISTRY_DIR.mkdir(
            parents=True,
            exist_ok=True
        )

        employee_dir = EMPLOYEE_REGISTRY_DIR / employee_id

        if employee_dir.exists():
            raise FileExistsError(
                f"Employee {employee_id} is already registered."
            )

        # 2. Create a temporary registration directory
        temporary_dir = (
            EMPLOYEE_REGISTRY_DIR
            / f".tmp_{uuid.uuid4().hex}"
        )

        reference_image_dir = temporary_dir / "reference_images"
        embedding_dir = temporary_dir / "embeddings"

        reference_image_dir.mkdir(parents=True)
        embedding_dir.mkdir(parents=True)

        try:
            valid_image_paths = []
            reference_names = []

            # 3. Validate and save uploaded reference photos
            for index, upload in enumerate(files, start=1):

                content_type = getattr(upload, "content_type", None)

                allowed_types = {
                    "image/jpeg": ".jpg",
                    "image/png": ".png",
                    "image/webp": ".webp"
                }

                if content_type not in allowed_types:
                    raise ValueError(
                        f"Reference photo {index}: "
                        "Only JPG, PNG and WEBP are allowed."
                    )

                image_bytes = upload.file.read(
                    settings.MAX_FILE_SIZE + 1
                )

                if not image_bytes:
                    raise ValueError(
                        f"Reference photo {index} is empty."
                    )

                if len(image_bytes) > settings.MAX_FILE_SIZE:
                    raise ValueError(
                        f"Reference photo {index} exceeds the size limit."
                    )

                image_array = np.frombuffer(
                    image_bytes,
                    dtype=np.uint8
                )

                image = cv2.imdecode(
                    image_array,
                    cv2.IMREAD_COLOR
                )

                if image is None:
                    raise ValueError(
                        f"Reference photo {index} is invalid."
                    )

                reference_name = f"reference_{index:03d}"
                extension = allowed_types[content_type]

                image_path = (
                    reference_image_dir
                    / f"{reference_name}{extension}"
                )

                image_path.write_bytes(image_bytes)

                # 4. Validate reference image quality
                quality = ImageQualityService.validate_crop(
                    image_path
                )

                if not quality["can_process"]:
                    raise ValueError(
                        f"Reference photo {index} is unsuitable: "
                        f"{quality.get('issues', [])}. "
                        "Use a clearer, larger person image."
                    )

                valid_image_paths.append(str(image_path))
                reference_names.append(reference_name)

            # 5. Extract OSNet embeddings using the existing model
            extractor = OSNetFeatureService.get_extractor()

            features = extractor(valid_image_paths)

            if isinstance(features, torch.Tensor):
                features = features.detach().cpu().numpy()

            features = np.asarray(features)

            if (
                features.ndim != 2
                or features.shape[0] != len(valid_image_paths)
            ):
                raise ValueError(
                    "Unexpected OSNet feature output."
                )

            if features.shape[1] != 512:
                raise ValueError(
                    f"Expected 512-dimensional OSNet embeddings, "
                    f"received {features.shape[1]}."
                )

            # 6. Normalize and save embeddings
            references = []

            for index, feature in enumerate(features):

                norm = np.linalg.norm(feature)

                if not np.isfinite(norm) or norm == 0:
                    raise ValueError(
                        f"Invalid embedding generated for photo {index + 1}."
                    )

                embedding = (
                    feature / norm
                ).astype(np.float32)

                embedding_filename = (
                    f"{reference_names[index]}.npy"
                )

                np.save(
                    embedding_dir / embedding_filename,
                    embedding
                )

                references.append({
                    "embedding_file": embedding_filename
                })

            # 7. Create employee metadata
            metadata = {
                "employee_id": employee_id,
                "employee_name": employee_name,
                "embedding_dimension": int(features.shape[1]),
                "total_reference_photos": len(references),
                "references": references
            }

            metadata_path = temporary_dir / "employee.json"

            with open(
                metadata_path,
                "w",
                encoding="utf-8"
            ) as file:
                json.dump(metadata, file, indent=4)

            # 8. Move completed registration into final location
            # Never expose a partially written employee directory.
            if employee_dir.exists():
                raise FileExistsError(
                    f"Employee {employee_id} is already registered."
                )

            temporary_dir.rename(employee_dir)

            return {
                "employee_id": employee_id,
                "employee_name": employee_name,
                "total_reference_photos": len(references),
                "embedding_dimension": int(features.shape[1]),
                "status": "registered"
            }

        except Exception:
            # Clean up incomplete registration
            if temporary_dir.exists():
                shutil.rmtree(temporary_dir)

            raise
