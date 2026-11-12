import numpy as np

from pathlib import Path

from app.services.osnet_service import EMBEDDING_DIR
from app.services.similarity_service import SimilarityService


UPLOAD_DIR = EMBEDDING_DIR.parent

VIDEO_EMBEDDING_DIR = UPLOAD_DIR / "video_embeddings"


class VideoMatchingService:

    @staticmethod
    def compare_with_reference(
        video_id: str,
        reference_image_id: str,
        interval_seconds: float = 1,
        threshold: float = 0.70,
        top_k: int = 3,
        min_score_gap: float = 0.05
    ):

        if interval_seconds <= 0:
            raise ValueError(
                "Interval must be greater than zero"
            )

        # -----------------------------------------
        # 1. Load reference-person embeddings
        # -----------------------------------------

        reference_folder = (
            EMBEDDING_DIR / str(reference_image_id)
        )

        if not reference_folder.exists():
            raise FileNotFoundError(
                "Reference image embeddings not found. "
                "Run image feature extraction first."
            )

        reference_files = sorted(
            reference_folder.glob("person_*.npy"),
            key=lambda path: int(
                path.stem.split("_")[1]
            )
        )

        if not reference_files:
            raise ValueError(
                "No reference-person embeddings found"
            )

        original_persons = []

        for embedding_path in reference_files:

            person_number = int(
                embedding_path.stem.split("_")[1]
            )

            embedding = np.load(
                embedding_path,
                allow_pickle=False
            )

            embedding = np.asarray(
                embedding,
                dtype=np.float32
            )

            if embedding.ndim != 1:
                raise ValueError(
                    f"Invalid reference embedding: "
                    f"{embedding_path.name}"
                )

            if not np.all(np.isfinite(embedding)):
                raise ValueError(
                    f"Reference embedding contains invalid values: "
                    f"{embedding_path.name}"
                )

            if np.linalg.norm(embedding) == 0:
                raise ValueError(
                    f"Reference embedding has zero magnitude: "
                    f"{embedding_path.name}"
                )

            original_persons.append({

                "person_number": person_number,

                "embedding": embedding

            })

        # -----------------------------------------
        # 2. Locate video embeddings
        # -----------------------------------------

        video_embedding_folder = (
            VIDEO_EMBEDDING_DIR / str(video_id)
        )

        if not video_embedding_folder.exists():
            raise FileNotFoundError(
                "Video embeddings not found. "
                "Run video feature extraction first."
            )

        frame_folders = sorted(
            [
                folder
                for folder in video_embedding_folder.iterdir()
                if folder.is_dir()
                and folder.name.startswith("frame_")
            ],
            key=lambda path: int(
                path.name.split("_")[1]
            )
        )

        if not frame_folders:
            raise ValueError(
                "No video frame embeddings found"
            )

        # -----------------------------------------
        # 3. Compare every video person
        # -----------------------------------------

        total_video_persons = 0
        total_compared_persons = 0
        total_matches_found = 0
        total_clear_matches = 0
        total_ambiguous_matches = 0

        frame_results = []

        for frame_folder in frame_folders:

            frame_number = int(
                frame_folder.name.split("_")[1]
            )

            timestamp_seconds = round(
                (frame_number - 1) * interval_seconds,
                3
            )

            video_embedding_files = sorted(
                frame_folder.glob("person_*.npy"),
                key=lambda path: int(
                    path.stem.split("_")[1]
                )
            )

            person_results = []

            for embedding_path in video_embedding_files:

                video_person_number = int(
                    embedding_path.stem.split("_")[1]
                )

                total_video_persons += 1

                search_embedding = np.load(
                    embedding_path,
                    allow_pickle=False
                )

                search_embedding = np.asarray(
                    search_embedding,
                    dtype=np.float32
                )

                if search_embedding.ndim != 1:
                    raise ValueError(
                        f"Invalid video embedding: "
                        f"{embedding_path.name}"
                    )

                if not np.all(np.isfinite(search_embedding)):
                    raise ValueError(
                        f"Video embedding contains invalid values: "
                        f"{embedding_path.name}"
                    )

                if np.linalg.norm(search_embedding) == 0:
                    raise ValueError(
                        f"Video embedding has zero magnitude: "
                        f"{embedding_path.name}"
                    )

                if (
                    search_embedding.shape[0]
                    != original_persons[0]["embedding"].shape[0]
                ):
                    raise ValueError(
                        "Reference and video embedding dimensions do not match"
                    )

                matching_result = (
                    SimilarityService.rank_matches(
                        search_embedding=search_embedding,
                        original_persons=original_persons,
                        threshold=threshold,
                        top_k=top_k,
                        min_score_gap=min_score_gap
                    )
                )

                total_compared_persons += 1

                if matching_result["match_found"]:
                    total_matches_found += 1

                if matching_result.get("match_status") == "clear_match":
                    total_clear_matches += 1

                elif matching_result.get("match_status") == "ambiguous":
                    total_ambiguous_matches += 1

                best_match = matching_result.get(
                    "best_match"
                )

                person_results.append({

                    "video_person_number": video_person_number,

                    "embedding_filename": embedding_path.name,

                    "match_found": matching_result["match_found"],

                    "match_status": matching_result.get(
                        "match_status"
                    ),

                    "best_match": best_match,

                    "top_matches": matching_result.get(
                        "top_matches",
                        []
                    ),

                    "best_similarity": matching_result.get(
                        "best_similarity"
                    ),

                    "second_best_similarity": matching_result.get(
                        "second_best_similarity"
                    ),

                    "score_gap": matching_result.get(
                        "score_gap"
                    )

                })

            frame_results.append({

                "frame_number": frame_number,

                "timestamp_seconds": timestamp_seconds,

                "total_video_persons": len(
                    video_embedding_files
                ),

                "person_results": person_results

            })

        # -----------------------------------------
        # 4. Return comparison results
        # -----------------------------------------

        return {

            "video_id": str(video_id),

            "reference_image_id": str(reference_image_id),

            "total_reference_persons": len(
                original_persons
            ),

            "total_frames_processed": len(
                frame_folders
            ),

            "total_video_persons": total_video_persons,

            "total_compared_persons": total_compared_persons,

            "total_matches_found": total_matches_found,

            "total_clear_matches": total_clear_matches,

            "total_ambiguous_matches": total_ambiguous_matches,

            "threshold_percentage": round(
                threshold * 100,
                2
            ),

            "minimum_score_gap_percentage": round(
                min_score_gap * 100,
                2
            ),

            "frames": frame_results

        }