
import json
import numpy as np

from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent

EMPLOYEE_REGISTRY_DIR = (
    BASE_DIR / "uploads" / "employee_registry"
)

VIDEO_TRACKING_DIR = (
    BASE_DIR / "uploads" / "video_tracking"
)

VIDEO_EMBEDDING_DIR = (
    BASE_DIR / "uploads" / "video_embeddings"
)

MATCH_THRESHOLD = 0.70
UNCERTAIN_THRESHOLD = 0.60
MINIMUM_SCORE_MARGIN = 0.05


class VideoEmployeeMatchingService:

    @staticmethod
    def _load_json(path: Path):

        if not path.exists():
            raise FileNotFoundError(
                f"Required metadata not found: {path}"
            )

        with open(path, "r", encoding="utf-8") as file:
            return json.load(file)

    @staticmethod
    def _load_employee_references():

        if not EMPLOYEE_REGISTRY_DIR.exists():
            raise ValueError(
                "Employee registry not found. "
                "Register employees before matching."
            )

        employees = []

        for metadata_path in sorted(
            EMPLOYEE_REGISTRY_DIR.glob("*/employee.json")
        ):

            with open(
                metadata_path,
                "r",
                encoding="utf-8"
            ) as file:
                metadata = json.load(file)

            employee_id = metadata["employee_id"]
            employee_name = metadata["employee_name"]

            embedding_dir = (
                metadata_path.parent / "embeddings"
            )

            references = []

            for reference in metadata.get("references", []):

                embedding_path = (
                    embedding_dir
                    / reference["embedding_file"]
                )

                if not embedding_path.exists():
                    continue

                embedding = np.asarray(
                    np.load(embedding_path),
                    dtype=np.float32
                )

                if (
                    embedding.ndim != 1
                    or not np.all(np.isfinite(embedding))
                ):
                    continue

                norm = np.linalg.norm(embedding)

                if norm == 0 or not np.isfinite(norm):
                    continue

                embedding = embedding / norm

                references.append(embedding)

            if references:
                employees.append({
                    "employee_id": employee_id,
                    "employee_name": employee_name,
                    "embedding_dimension": metadata.get(
                        "embedding_dimension"
                    ),
                    "references": references
                })

        if not employees:
            raise ValueError(
                "No valid employee reference embeddings found."
            )

        dimensions = {
            len(reference)
            for employee in employees
            for reference in employee["references"]
        }

        if len(dimensions) != 1:
            raise ValueError(
                "Employee reference embeddings have inconsistent dimensions."
            )

        return employees

    @staticmethod
    def _cosine_similarity(first, second):

        first = np.asarray(first, dtype=np.float32)
        second = np.asarray(second, dtype=np.float32)

        if first.shape != second.shape:
            raise ValueError(
                "Embedding dimensions do not match."
            )

        first_norm = np.linalg.norm(first)
        second_norm = np.linalg.norm(second)

        if first_norm == 0 or second_norm == 0:
            return None

        return float(
            np.dot(first, second)
            / (first_norm * second_norm)
        )

    @classmethod
    def _get_track_embeddings(
        cls,
        video_id: str,
        tracking_metadata: dict
    ):

        track_embeddings = {}

        for frame in tracking_metadata.get("frames", []):

            frame_filename = frame["frame_filename"]
            frame_stem = Path(frame_filename).stem

            for person in frame.get("persons", []):

                track_id = person.get("track_id")
                person_number = person.get("person_number")

                if not track_id or person_number is None:
                    continue

                embedding_path = (
                    VIDEO_EMBEDDING_DIR
                    / video_id
                    / frame_stem
                    / f"person_{person_number}.npy"
                )

                if not embedding_path.exists():
                    continue

                embedding = np.asarray(
                    np.load(embedding_path),
                    dtype=np.float32
                )

                if (
                    embedding.ndim != 1
                    or not np.all(np.isfinite(embedding))
                ):
                    continue

                norm = np.linalg.norm(embedding)

                if norm == 0 or not np.isfinite(norm):
                    continue

                embedding = embedding / norm

                track_embeddings.setdefault(
                    track_id, []
                ).append({
                    "frame_number": frame.get("frame_number"),
                    "timestamp_seconds": frame.get(
                        "timestamp_seconds"
                    ),
                    "person_number": person_number,
                    "embedding": embedding
                })

        return track_embeddings

    @classmethod
    def _compare_track(
        cls,
        track_id: str,
        observations: list,
        employees: list
    ):

        employee_scores = []

        for employee in employees:

            observation_scores = []

            for observation in observations:

                similarities = []

                for reference in employee["references"]:

                    similarity = cls._cosine_similarity(
                        observation["embedding"],
                        reference
                    )

                    if similarity is not None:
                        similarities.append(similarity)

                if similarities:
                    observation_scores.append(
                        max(similarities)
                    )

            if not observation_scores:
                continue

            # Average the best three frame-level scores.
            # This reduces dependence on a single frame.
            best_scores = sorted(
                observation_scores,
                reverse=True
            )[:3]

            aggregate_score = float(
                np.mean(best_scores)
            )

            employee_scores.append({
                "employee_id": employee["employee_id"],
                "employee_name": employee["employee_name"],
                "similarity_score": round(
                    aggregate_score, 6
                ),
                "frames_compared": len(observation_scores)
            })

        employee_scores.sort(
            key=lambda item: item["similarity_score"],
            reverse=True
        )

        if not employee_scores:
            return {
                "track_id": track_id,
                "match_status": "unknown",
                "best_match": None,
                "top_matches": [],
                "frames_with_embeddings": len(observations)
            }

        best_match = employee_scores[0]

        second_score = (
            employee_scores[1]["similarity_score"]
            if len(employee_scores) > 1
            else 0.0
        )

        score = best_match["similarity_score"]
        margin = score - second_score

        if (
            score >= MATCH_THRESHOLD
            and margin >= MINIMUM_SCORE_MARGIN
        ):
            status = "employee_match"

        elif score >= UNCERTAIN_THRESHOLD:
            status = "uncertain"

        else:
            status = "unknown"

        return {
            "track_id": track_id,
            "match_status": status,
            "best_match": best_match,
            "score_margin": round(margin, 6),
            "top_matches": employee_scores[:3],
            "frames_with_embeddings": len(observations)
        }

    @classmethod
    def match_video(cls, video_id: str):

        tracking_metadata_path = (
            VIDEO_TRACKING_DIR
            / video_id
            / "tracking_metadata.json"
        )

        tracking_metadata = cls._load_json(
            tracking_metadata_path
        )

        employees = cls._load_employee_references()

        track_embeddings = cls._get_track_embeddings(
            video_id,
            tracking_metadata
        )

        results = []

        for track_id, observations in sorted(
            track_embeddings.items()
        ):

            result = cls._compare_track(
                track_id=track_id,
                observations=observations,
                employees=employees
            )

            results.append(result)

        matched = [
            item for item in results
            if item["match_status"] == "employee_match"
        ]

        unknown = [
            item for item in results
            if item["match_status"] == "unknown"
        ]

        uncertain = [
            item for item in results
            if item["match_status"] == "uncertain"
        ]

        output_dir = (
            VIDEO_TRACKING_DIR / video_id
        )

        output_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        output_path = (
            output_dir / "employee_matching_metadata.json"
        )

        report = {
            "video_id": video_id,
            "total_registered_employees": len(employees),
            "total_tracks_with_embeddings": len(results),
            "total_employee_matches": len(matched),
            "total_unknown_tracks": len(unknown),
            "total_uncertain_tracks": len(uncertain),
            "thresholds": {
                "match_threshold": MATCH_THRESHOLD,
                "uncertain_threshold": UNCERTAIN_THRESHOLD,
                "minimum_score_margin": MINIMUM_SCORE_MARGIN
            },
            "results": results
        }

        temporary_path = output_path.with_suffix(".tmp")

        with open(
            temporary_path,
            "w",
            encoding="utf-8"
        ) as file:
            json.dump(report, file, indent=4)

        temporary_path.replace(output_path)

        report["report_path"] = str(output_path)

        return report
