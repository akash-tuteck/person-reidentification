
import json
import shutil

from pathlib import Path

import cv2
import numpy as np

from app.services.osnet_service import EMBEDDING_DIR


BASE_DIR = Path(__file__).resolve().parent.parent
UPLOAD_DIR = BASE_DIR / "uploads"

FRAME_DIR = UPLOAD_DIR / "frames"
DETECTION_DIR = UPLOAD_DIR / "video_detections"
EMBEDDING_BASE_DIR = EMBEDDING_DIR.parent / "video_embeddings"

TRACKING_DIR = UPLOAD_DIR / "video_tracking"


class VideoTrackingService:

    # A track can remain unmatched for this many sampled frames.
    MAX_FRAME_GAP = 2

    # Initial association thresholds; tune these using your own videos.
    MIN_APPEARANCE_SIMILARITY = 0.55
    MAX_NORMALIZED_DISTANCE = 0.12
    MIN_IOU_FOR_SPATIAL_MATCH = 0.15

    @staticmethod
    def _read_json(path):

        with path.open("r", encoding="utf-8") as file:
            return json.load(file)

    @staticmethod
    def _get_bbox(detection):

        bbox = (
            detection.get("bbox")
            or detection.get("bounding_box")
            or detection.get("box")
        )

        if isinstance(bbox, dict):

            if all(key in bbox for key in ("x1", "y1", "x2", "y2")):
                return [
                    float(bbox["x1"]),
                    float(bbox["y1"]),
                    float(bbox["x2"]),
                    float(bbox["y2"])
                ]

            if all(key in bbox for key in ("x", "y", "width", "height")):
                x = float(bbox["x"])
                y = float(bbox["y"])
                return [
                    x,
                    y,
                    x + float(bbox["width"]),
                    y + float(bbox["height"])
                ]

        if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
            return [float(value) for value in bbox]

        raise ValueError(
            "Person detection is missing a valid bounding box"
        )

    @staticmethod
    def _calculate_iou(box_a, box_b):

        x_left = max(box_a[0], box_b[0])
        y_top = max(box_a[1], box_b[1])
        x_right = min(box_a[2], box_b[2])
        y_bottom = min(box_a[3], box_b[3])

        intersection_width = max(0.0, x_right - x_left)
        intersection_height = max(0.0, y_bottom - y_top)

        intersection = intersection_width * intersection_height

        area_a = max(0.0, box_a[2] - box_a[0]) * max(
            0.0, box_a[3] - box_a[1]
        )

        area_b = max(0.0, box_b[2] - box_b[0]) * max(
            0.0, box_b[3] - box_b[1]
        )

        union = area_a + area_b - intersection

        if union <= 0:
            return 0.0

        return intersection / union

    @staticmethod
    def _center(box):

        return (
            (box[0] + box[2]) / 2.0,
            (box[1] + box[3]) / 2.0
        )

    @staticmethod
    def _cosine_similarity(embedding_a, embedding_b):

        if embedding_a is None or embedding_b is None:
            return None

        if embedding_a.shape != embedding_b.shape:
            return None

        norm_a = np.linalg.norm(embedding_a)
        norm_b = np.linalg.norm(embedding_b)

        if norm_a == 0 or norm_b == 0:
            return None

        return float(
            np.dot(embedding_a, embedding_b) / (norm_a * norm_b)
        )

    @staticmethod
    def _load_embedding(path):

        if not path.exists():
            return None

        try:

            embedding = np.load(path).astype(np.float32)

            if (
                embedding.ndim != 1
                or not np.all(np.isfinite(embedding))
                or np.linalg.norm(embedding) == 0
            ):
                return None

            return embedding

        except (OSError, ValueError):
            return None

    @classmethod
    def _association_score(
        cls,
        track,
        detection,
        frame_width,
        frame_height
    ):

        previous_box = track["last_bbox"]
        current_box = detection["bbox"]

        iou = cls._calculate_iou(previous_box, current_box)

        previous_center = cls._center(previous_box)
        current_center = cls._center(current_box)

        distance = np.sqrt(
            (previous_center[0] - current_center[0]) ** 2
            + (previous_center[1] - current_center[1]) ** 2
        )

        frame_diagonal = np.sqrt(
            frame_width ** 2 + frame_height ** 2
        )

        normalized_distance = (
            distance / frame_diagonal
            if frame_diagonal > 0
            else 1.0
        )

        similarity = cls._cosine_similarity(
            track["last_embedding"],
            detection["embedding"]
        )

        spatial_match = (
            iou >= cls.MIN_IOU_FOR_SPATIAL_MATCH
            or normalized_distance <= cls.MAX_NORMALIZED_DISTANCE
        )

        appearance_match = (
            similarity is not None
            and similarity >= cls.MIN_APPEARANCE_SIMILARITY
            and normalized_distance <= 0.30
        )

        if not spatial_match and not appearance_match:
            return None

        spatial_score = max(
            iou,
            max(0.0, 1.0 - normalized_distance / 0.30)
        )

        if similarity is not None:

            score = (
                0.65 * max(0.0, similarity)
                + 0.35 * spatial_score
            )

        else:
            score = spatial_score

        return float(score)

    @classmethod
    def track_persons(cls, video_id):

        video_id = str(video_id)

        frame_metadata_path = (
            FRAME_DIR / video_id / "frames_metadata.json"
        )

        detection_metadata_path = (
            DETECTION_DIR / video_id / "detection_metadata.json"
        )

        if not frame_metadata_path.exists():
            raise FileNotFoundError(
                "Extract video frames before tracking"
            )

        if not detection_metadata_path.exists():
            raise FileNotFoundError(
                "Detect persons in the video before tracking"
            )

        frame_metadata = cls._read_json(frame_metadata_path)
        detection_metadata = cls._read_json(detection_metadata_path)

        frame_lookup = {
            Path(frame["frame_filename"]).stem: frame
            for frame in frame_metadata.get("frames", [])
            if frame.get("frame_filename")
        }

        detection_frames = detection_metadata.get("frames", [])

        if not detection_frames:
            raise ValueError(
                "No detection frames found. Run person detection first."
            )

        output_dir = TRACKING_DIR / video_id
        tracked_frames_dir = output_dir / "tracked_frames"

        if output_dir.exists():
            shutil.rmtree(output_dir)

        tracked_frames_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        tracks = {}
        frame_results = []

        next_track_id = 1
        total_detections = 0
        total_matched_detections = 0

        detection_frames = sorted(
            detection_frames,
            key=lambda item: int(item.get("frame_number", 0))
        )

        for frame_data in detection_frames:

            frame_number = int(frame_data["frame_number"])

            frame_filename = (
                frame_data.get("frame_filename")
                or frame_data.get("frame_name")
            )

            if not frame_filename:
                raise ValueError(
                    f"Missing frame filename for frame {frame_number}"
                )

            frame_stem = Path(frame_filename).stem

            source_frame = frame_lookup.get(frame_stem)

            if source_frame is None:
                raise ValueError(
                    f"Frame metadata not found for {frame_stem}"
                )

            timestamp = float(
                source_frame.get(
                    "timestamp_seconds",
                    frame_data.get("timestamp_seconds", 0)
                )
            )

            frame_width = int(
                source_frame.get("frame_width", 0)
            )

            frame_height = int(
                source_frame.get("frame_height", 0)
            )

            detections = (
                frame_data.get("detections")
                or frame_data.get("persons")
                or []
            )

            frame_detections = []

            for detection in detections:

                person_number = detection.get("person_number")

                if person_number is None:
                    raise ValueError(
                        f"Missing person_number in {frame_stem}"
                    )

                bbox = cls._get_bbox(detection)

                embedding_path = (
                    EMBEDDING_BASE_DIR
                    / video_id
                    / frame_stem
                    / f"person_{person_number}.npy"
                )

                frame_detections.append({
                    "person_number": int(person_number),
                    "bbox": bbox,
                    "confidence": detection.get("confidence"),
                    "embedding": cls._load_embedding(embedding_path)
                })

            # Only tracks seen recently can receive a new detection.
            active_tracks = {
                track_id: track
                for track_id, track in tracks.items()
                if frame_number - track["last_frame_number"]
                <= cls.MAX_FRAME_GAP + 1
            }

            candidate_pairs = []

            for detection_index, detection in enumerate(frame_detections):

                for track_id, track in active_tracks.items():

                    score = cls._association_score(
                        track,
                        detection,
                        frame_width,
                        frame_height
                    )

                    if score is not None:
                        candidate_pairs.append(
                            (score, detection_index, track_id)
                        )

            # Highest scoring association is selected first.
            candidate_pairs.sort(
                key=lambda item: item[0],
                reverse=True
            )

            assigned_detections = set()
            assigned_tracks = set()

            for score, detection_index, track_id in candidate_pairs:

                if detection_index in assigned_detections:
                    continue

                if track_id in assigned_tracks:
                    continue

                detection = frame_detections[detection_index]
                track = tracks[track_id]

                track["last_bbox"] = detection["bbox"]
                track["last_frame_number"] = frame_number
                track["last_timestamp_seconds"] = timestamp
                track["last_embedding"] = (
                    detection["embedding"]
                    if detection["embedding"] is not None
                    else track["last_embedding"]
                )

                track["detection_count"] += 1
                track["frames"].append(frame_number)

                detection["track_id"] = track_id
                detection["association_score"] = round(score, 4)

                assigned_detections.add(detection_index)
                assigned_tracks.add(track_id)

                total_matched_detections += 1

            # Unassigned detections start new tracks.
            for detection_index, detection in enumerate(frame_detections):

                if detection_index in assigned_detections:
                    continue

                track_id = f"track_{next_track_id:05d}"
                next_track_id += 1

                tracks[track_id] = {
                    "track_id": track_id,
                    "first_frame_number": frame_number,
                    "last_frame_number": frame_number,
                    "first_timestamp_seconds": timestamp,
                    "last_timestamp_seconds": timestamp,
                    "last_bbox": detection["bbox"],
                    "last_embedding": detection["embedding"],
                    "detection_count": 1,
                    "frames": [frame_number]
                }

                detection["track_id"] = track_id
                detection["association_score"] = None

            total_detections += len(frame_detections)

            # Annotate the original extracted frame.
            original_frame_path = (
                FRAME_DIR / video_id / frame_filename
            )

            image = cv2.imread(str(original_frame_path))

            if image is None:
                raise IOError(
                    f"Could not read frame: {original_frame_path}"
                )

            for detection in frame_detections:

                x1, y1, x2, y2 = [
                    int(value) for value in detection["bbox"]
                ]

                track_id = detection["track_id"]

                cv2.rectangle(
                    image,
                    (x1, y1),
                    (x2, y2),
                    (0, 255, 0),
                    2
                )

                cv2.putText(
                    image,
                    track_id,
                    (x1, max(20, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2
                )

            tracked_frame_path = (
                tracked_frames_dir / frame_filename
            )

            if not cv2.imwrite(
                str(tracked_frame_path),
                image
            ):
                raise IOError(
                    f"Could not save tracked frame: {tracked_frame_path}"
                )

            frame_results.append({
                "frame_number": frame_number,
                "frame_filename": frame_filename,
                "timestamp_seconds": timestamp,
                "total_detections": len(frame_detections),
                "persons": [
                    {
                        "person_number": item["person_number"],
                        "track_id": item["track_id"],
                        "bbox": item["bbox"],
                        "confidence": item["confidence"],
                        "association_score": item["association_score"],
                        "has_embedding": item["embedding"] is not None
                    }
                    for item in frame_detections
                ],
                "tracked_frame_path": str(tracked_frame_path)
            })

        track_summaries = []

        for track_id, track in tracks.items():

            track_summaries.append({
                "track_id": track_id,
                "first_frame_number": track["first_frame_number"],
                "last_frame_number": track["last_frame_number"],
                "first_timestamp_seconds": track[
                    "first_timestamp_seconds"
                ],
                "last_timestamp_seconds": track[
                    "last_timestamp_seconds"
                ],
                "total_detections": track["detection_count"],
                "frames": track["frames"]
            })

        result = {
            "video_id": video_id,
            "total_frames_processed": len(frame_results),
            "total_person_detections": total_detections,
            "total_tracks_created": len(track_summaries),
            "detections_assigned_to_existing_tracks": (
                total_matched_detections
            ),
            "tracking_folder": str(output_dir),
            "tracked_frames_folder": str(tracked_frames_dir),
            "tracks": track_summaries,
            "frames": frame_results
        }

        metadata_path = output_dir / "tracking_metadata.json"

        with metadata_path.open("w", encoding="utf-8") as file:
            json.dump(result, file, indent=4)

        result["metadata_path"] = str(metadata_path)

        return result
