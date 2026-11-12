
import cv2
import numpy as np
import torch

from pathlib import Path

from app.services.osnet_service import (
    OSNetFeatureService,
    EMBEDDING_DIR
)

from app.services.image_quality_service import (
    ImageQualityService
)


UPLOAD_DIR = EMBEDDING_DIR.parent

VIDEO_PERSON_DIR = UPLOAD_DIR / "video_persons"

VIDEO_EMBEDDING_DIR = UPLOAD_DIR / "video_embeddings"


class VideoOSNetFeatureService:

    @classmethod
    def extract_features(cls, video_id, interval_seconds=1):

        if interval_seconds <= 0:
            raise ValueError(
                "Interval must be greater than zero"
            )

        video_person_folder = (
            VIDEO_PERSON_DIR / str(video_id)
        )

        if not video_person_folder.exists():
            raise FileNotFoundError(
                "Video person crops not found. "
                "Run video person detection first."
            )

        frame_folders = sorted(
            [
                folder
                for folder in video_person_folder.iterdir()
                if folder.is_dir()
                and folder.name.startswith("frame_")
            ],
            key=lambda path: int(
                path.name.split("_")[1]
            )
        )

        if not frame_folders:
            raise ValueError(
                "No frame person folders found"
            )

        video_embedding_folder = (
            VIDEO_EMBEDDING_DIR / str(video_id)
        )

        video_embedding_folder.mkdir(
            parents=True,
            exist_ok=True
        )

        # Reuse the existing OSNet model.
        # The model is loaded only once.
        extractor = OSNetFeatureService.get_extractor()

        total_detected_persons = 0
        total_processed_persons = 0
        total_skipped_persons = 0

        embedding_dimension = None

        frame_results = []

        for frame_folder in frame_folders:

            frame_number = int(
                frame_folder.name.split("_")[1]
            )

            timestamp_seconds = round(
                (frame_number - 1) * interval_seconds,
                3
            )

            person_images = sorted(
                frame_folder.glob("person_*.jpg"),
                key=lambda path: int(
                    path.stem.split("_")[1]
                )
            )

            frame_embedding_folder = (
                video_embedding_folder
                / frame_folder.name
            )

            frame_embedding_folder.mkdir(
                parents=True,
                exist_ok=True
            )

            # Remove old embeddings for this frame
            for old_file in frame_embedding_folder.glob(
                "person_*.npy"
            ):
                old_file.unlink()

            total_detected_persons += len(person_images)

            valid_persons = []
            skipped_persons = []

            # 1. Validate person crops
            for image_path in person_images:

                person_number = int(
                    image_path.stem.split("_")[1]
                )

                quality_result = (
                    ImageQualityService.validate_crop(
                        image_path
                    )
                )

                if quality_result["can_process"]:

                    valid_persons.append({
                        "person_number": person_number,
                        "image_path": image_path,
                        "quality": quality_result
                    })

                else:

                    skipped_persons.append({
                        "person_number": person_number,
                        "quality": quality_result
                    })

            # 2. Extract OSNet features for valid crops
            processed_persons = []

            if valid_persons:

                image_paths = [
                    str(person["image_path"])
                    for person in valid_persons
                ]

                features = extractor(image_paths)

                if isinstance(features, torch.Tensor):

                    features = (
                        features.detach()
                        .cpu()
                        .numpy()
                    )

                features = np.asarray(features)

                if (
                    features.ndim != 2
                    or features.shape[0] != len(valid_persons)
                ):
                    raise ValueError(
                        f"Unexpected OSNet output shape "
                        f"for {frame_folder.name}: "
                        f"{features.shape}"
                    )

                embedding_dimension = int(
                    features.shape[1]
                )

                # 3. Normalize and save embeddings
                for index, feature in enumerate(features):

                    if not np.all(np.isfinite(feature)):
                        raise ValueError(
                            "OSNet generated non-finite values"
                        )

                    norm = np.linalg.norm(feature)

                    if not np.isfinite(norm) or norm == 0:
                        raise ValueError(
                            "OSNet generated an invalid embedding"
                        )

                    normalized_feature = (
                        feature / norm
                    ).astype(np.float32)

                    person = valid_persons[index]

                    person_number = person["person_number"]

                    embedding_path = (
                        frame_embedding_folder
                        / f"person_{person_number}.npy"
                    )

                    np.save(
                        embedding_path,
                        normalized_feature
                    )

                    processed_persons.append({

                        "person_number": person_number,

                        "embedding_dimension": int(
                            len(normalized_feature)
                        ),

                        "embedding_path": str(
                            embedding_path
                        ),

                        "quality": person["quality"]

                    })

            processed_count = len(processed_persons)
            skipped_count = len(skipped_persons)

            total_processed_persons += processed_count
            total_skipped_persons += skipped_count

            frame_results.append({

                "frame_number": frame_number,

                "frame_filename": frame_folder.name,

                "timestamp_seconds": timestamp_seconds,

                "total_detected_persons": len(person_images),

                "total_processed_persons": processed_count,

                "total_skipped_persons": skipped_count,

                "embedding_folder": str(
                    frame_embedding_folder
                ),

                "persons": processed_persons,

                "skipped_persons": skipped_persons

            })

        return {

            "video_id": str(video_id),

            "total_frames_processed": len(frame_folders),

            "total_detected_persons": total_detected_persons,

            "total_processed_persons": total_processed_persons,

            "total_skipped_persons": total_skipped_persons,

            "embedding_dimension": embedding_dimension,

            "embedding_folder": str(
                video_embedding_folder
            ),

            "frames": frame_results

        }
