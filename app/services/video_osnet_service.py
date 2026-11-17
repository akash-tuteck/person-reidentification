
import json
import shutil
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

FRAME_DIR = UPLOAD_DIR / "frames"
VIDEO_PERSON_DIR = UPLOAD_DIR / "video_persons"
VIDEO_DETECTION_DIR = UPLOAD_DIR / "video_detections"
VIDEO_EMBEDDING_DIR = UPLOAD_DIR / "video_embeddings"


class VideoOSNetFeatureService:

    @classmethod
    def extract_features(cls, video_id, interval_seconds=1):

        if interval_seconds <= 0:
            raise ValueError(
                "Interval must be greater than zero"
            )

        video_id = str(video_id)

        video_person_folder = VIDEO_PERSON_DIR / video_id
        frame_folder = FRAME_DIR / video_id
        detection_folder = VIDEO_DETECTION_DIR / video_id

        if not video_person_folder.exists():
            raise FileNotFoundError(
                "Video person crops not found. "
                "Run video person detection first."
            )

        # Load the original frame metadata.
        frame_metadata_path = (
            frame_folder / "frames_metadata.json"
        )

        if not frame_metadata_path.exists():
            raise FileNotFoundError(
                "Frame metadata not found. Extract frames again."
            )

        with open(
            frame_metadata_path,
            "r",
            encoding="utf-8"
        ) as file:
            frame_metadata = json.load(file)

        # Load detection metadata.
        detection_metadata_path = (
            detection_folder / "detection_metadata.json"
        )

        if not detection_metadata_path.exists():
            raise FileNotFoundError(
                "Detection metadata not found. "
                "Run person detection first."
            )

        with open(
            detection_metadata_path,
            "r",
            encoding="utf-8"
        ) as file:
            detection_metadata = json.load(file)

        # Use only frames present in current detection metadata.
        # This excludes stale folders and duplicate frame names.
        detection_frames = detection_metadata.get("frames", [])

        if not detection_frames:
            raise ValueError(
                "No processed frames found in detection metadata"
            )

        # Map actual timestamps from frame extraction metadata.
        timestamp_lookup = {
            item["frame_filename"]: item["timestamp_seconds"]
            for item in frame_metadata.get("frames", [])
        }

        video_embedding_folder = (
            VIDEO_EMBEDDING_DIR / video_id
        )

        # Remove all old embeddings before generating new ones.
        if video_embedding_folder.exists():
            shutil.rmtree(video_embedding_folder)

        video_embedding_folder.mkdir(
            parents=True,
            exist_ok=True
        )

        # Reuse existing OSNet model.
        extractor = OSNetFeatureService.get_extractor()

        total_detected_persons = 0
        total_processed_persons = 0
        total_skipped_persons = 0

        embedding_dimension = None
        frame_results = []

        for detection_frame in detection_frames:

            frame_filename = detection_frame["frame_filename"]
            frame_stem = Path(frame_filename).stem

            frame_number = detection_frame["frame_number"]

            timestamp_seconds = timestamp_lookup.get(
                frame_filename,
                detection_frame.get("timestamp_seconds")
            )

            if timestamp_seconds is None:
                raise ValueError(
                    f"Timestamp missing for {frame_filename}"
                )

            frame_person_folder = (
                video_person_folder / frame_stem
            )

            if not frame_person_folder.exists():
                raise FileNotFoundError(
                    f"Person crop folder missing: {frame_stem}"
                )

            person_images = sorted(
                frame_person_folder.glob("person_*.jpg"),
                key=lambda path: int(path.stem.split("_")[1])
            )

            frame_embedding_folder = (
                video_embedding_folder / frame_stem
            )

            frame_embedding_folder.mkdir(
                parents=True,
                exist_ok=True
            )

            total_detected_persons += len(person_images)

            valid_persons = []
            skipped_persons = []

            # 1. Validate person crops.
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

            # 2. Extract OSNet features.
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
                        f"for {frame_stem}: {features.shape}"
                    )

                embedding_dimension = int(features.shape[1])

                # 3. Normalize and save embeddings.
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
                        "embedding_path": str(embedding_path),
                        "quality": person["quality"]
                    })

            processed_count = len(processed_persons)
            skipped_count = len(skipped_persons)

            total_processed_persons += processed_count
            total_skipped_persons += skipped_count

            frame_results.append({
                "frame_number": frame_number,
                "frame_filename": frame_stem,
                "timestamp_seconds": timestamp_seconds,
                "total_detected_persons": len(person_images),
                "total_processed_persons": processed_count,
                "total_skipped_persons": skipped_count,
                "embedding_folder": str(frame_embedding_folder),
                "persons": processed_persons,
                "skipped_persons": skipped_persons
            })

        return {
            "video_id": video_id,
            "total_frames_processed": len(frame_results),
            "total_detected_persons": total_detected_persons,
            "total_processed_persons": total_processed_persons,
            "total_skipped_persons": total_skipped_persons,
            "embedding_dimension": embedding_dimension,
            "embedding_folder": str(video_embedding_folder),
            "frames": frame_results
        }
