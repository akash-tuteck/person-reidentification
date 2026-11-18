
import json
import numpy as np

from pathlib import Path

from app.services.object_feature_service import (
    ObjectFeatureService
)

from app.services.object_categories import (
    NON_PERSON_COCO_CLASS_IDS
)


BASE_DIR = Path(__file__).resolve().parent.parent

UPLOAD_DIR = BASE_DIR / "uploads"

VIDEO_OBJECT_DIR = UPLOAD_DIR / "video_objects"

VIDEO_EMBEDDING_DIR = (
    UPLOAD_DIR / "video_embeddings" / "objects"
)

VIDEO_DETECTION_DIR = (
    UPLOAD_DIR / "video_object_detections"
)


class VideoObjectFeatureService:

    MODEL_NAME = ObjectFeatureService.MODEL_NAME

    EMBEDDING_DIMENSION = (
        ObjectFeatureService.EMBEDDING_DIMENSION
    )

    SUPPORTED_CATEGORIES = set(
        NON_PERSON_COCO_CLASS_IDS.keys()
    )

    @classmethod
    def extract_features(cls, video_id: str) -> dict:
        """
        Extract DINOv2 embeddings for all detected
        non-human objects across extracted video frames.
        """

        video_id = str(video_id)

        video_object_dir = VIDEO_OBJECT_DIR / video_id

        if not video_object_dir.exists():
            raise ValueError(
                f"Video object detections not found: {video_id}"
            )

        embedding_video_dir = (
            VIDEO_EMBEDDING_DIR / video_id
        )

        embedding_video_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        total_processed = 0

        category_counts = {}

        frame_results = []

        processed_objects = []

        # Process each extracted frame.
        frame_directories = sorted(
            path for path in video_object_dir.iterdir()
            if path.is_dir()
        )

        for frame_dir in frame_directories:

            frame_name = frame_dir.name

            frame_embedding_dir = (
                embedding_video_dir / frame_name
            )

            frame_embedding_dir.mkdir(
                parents=True,
                exist_ok=True
            )

            frame_total = 0

            frame_category_counts = {}

            # Process each category.
            for category_dir in sorted(
                path for path in frame_dir.iterdir()
                if path.is_dir()
            ):

                category = category_dir.name

                # Reject person and unsupported categories.
                if category not in cls.SUPPORTED_CATEGORIES:
                    continue

                crop_files = sorted(
                    category_dir.glob("*.jpg")
                )

                if not crop_files:
                    continue

                category_embedding_dir = (
                    frame_embedding_dir / category
                )

                category_embedding_dir.mkdir(
                    parents=True,
                    exist_ok=True
                )

                category_count = 0

                for crop_path in crop_files:

                    embedding = (
                        ObjectFeatureService.extract_embedding(
                            crop_path
                        )
                    )

                    embedding_path = (
                        category_embedding_dir /
                        f"{crop_path.stem}.npy"
                    )

                    np.save(
                        embedding_path,
                        embedding
                    )

                    category_count += 1
                    frame_total += 1
                    total_processed += 1

                    processed_object = {
                        "frame": frame_name,
                        "category": category,
                        "object_filename": crop_path.name,
                        "embedding_filename": embedding_path.name,
                        "embedding_path": str(embedding_path),
                        "embedding_dimension": int(
                            embedding.shape[0]
                        )
                    }

                    processed_objects.append(
                        processed_object
                    )

                if category_count > 0:

                    frame_category_counts[category] = (
                        category_count
                    )

                    category_counts[category] = (
                        category_counts.get(category, 0)
                        + category_count
                    )

            if frame_total > 0:

                frame_results.append({
                    "frame": frame_name,
                    "total_objects_processed": frame_total,
                    "category_counts": frame_category_counts
                })

        output = {
            "video_id": video_id,
            "total_frames_processed": len(frame_results),
            "total_objects_processed": total_processed,
            "category_counts": category_counts,
            "embedding_dimension": cls.EMBEDDING_DIMENSION,
            "model": cls.MODEL_NAME,
            "device": str(ObjectFeatureService._device),
            "status": (
                "completed"
                if total_processed > 0
                else "no_objects_found"
            ),
            "embedding_folder": str(
                embedding_video_dir
            ),
            "frames": frame_results,
            "processed_objects": processed_objects
        }

        metadata_dir = VIDEO_DETECTION_DIR / video_id

        metadata_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        metadata_path = (
            metadata_dir / "object_feature_metadata.json"
        )

        with metadata_path.open(
            "w",
            encoding="utf-8"
        ) as file:

            json.dump(
                output,
                file,
                indent=4
            )

        output["metadata_path"] = str(metadata_path)

        return output
