
import json
from pathlib import Path

import cv2
import numpy as np


BASE_DIR = Path(__file__).resolve().parent.parent
UPLOAD_DIR = BASE_DIR / "uploads"

FRAME_DIR = UPLOAD_DIR / "frames"
OBJECT_DETECTION_DIR = UPLOAD_DIR / "video_object_detections"
OBJECT_EMBEDDING_DIR = UPLOAD_DIR / "video_embeddings" / "objects"
OBJECT_TRACKING_DIR = UPLOAD_DIR / "video_object_tracking"


class VideoObjectTrackingService:
    """
    Tracks detected non-human objects across the extracted video frames.

    Association is category-aware and uses:
      - bounding-box IoU
      - normalized center distance
      - DINOv2 embedding cosine similarity

    Track IDs are local to one video.
    """

    MAX_FRAME_GAP = 2

    # Initial thresholds; tune against your CCTV test videos.
    MIN_APPEARANCE_SIMILARITY = 0.55
    MAX_NORMALIZED_DISTANCE = 0.15
    MIN_IOU_FOR_SPATIAL_MATCH = 0.05

    @staticmethod
    def _read_json(path):
        with open(path, "r", encoding="utf-8") as file:
            return json.load(file)

    @staticmethod
    def _write_json(path, data):
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w", encoding="utf-8") as file:
            json.dump(data, file, indent=4)

    @staticmethod
    def _get_value(item, *keys, default=None):
        for key in keys:
            if key in item and item[key] is not None:
                return item[key]
        return default

    @classmethod
    def _get_bbox(cls, detection):
        bbox = cls._get_value(
            detection,
            "bbox",
            "bounding_box",
            "box"
        )

        if isinstance(bbox, dict):
            if all(k in bbox for k in ("x1", "y1", "x2", "y2")):
                return [
                    float(bbox["x1"]),
                    float(bbox["y1"]),
                    float(bbox["x2"]),
                    float(bbox["y2"])
                ]

            if all(k in bbox for k in ("x", "y", "width", "height")):
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

        return None

    @staticmethod
    def _calculate_iou(box_a, box_b):
        ax1, ay1, ax2, ay2 = box_a
        bx1, by1, bx2, by2 = box_b

        x1 = max(ax1, bx1)
        y1 = max(ay1, by1)
        x2 = min(ax2, bx2)
        y2 = min(ay2, by2)

        intersection_width = max(0.0, x2 - x1)
        intersection_height = max(0.0, y2 - y1)
        intersection = intersection_width * intersection_height

        area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
        area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)

        union = area_a + area_b - intersection

        if union <= 0:
            return 0.0

        return intersection / union

    @staticmethod
    def _center(box):
        x1, y1, x2, y2 = box
        return (
            (x1 + x2) / 2.0,
            (y1 + y2) / 2.0
        )

    @staticmethod
    def _cosine_similarity(embedding_a, embedding_b):
        if embedding_a is None or embedding_b is None:
            return None

        a = np.asarray(embedding_a, dtype=np.float32).reshape(-1)
        b = np.asarray(embedding_b, dtype=np.float32).reshape(-1)

        if a.shape != b.shape:
            return None

        denominator = np.linalg.norm(a) * np.linalg.norm(b)

        if denominator == 0:
            return None

        return float(np.dot(a, b) / denominator)

    @staticmethod
    def _load_embedding(path):
        if not path or not Path(path).exists():
            return None

        try:
            embedding = np.load(path).astype(np.float32).reshape(-1)

            if embedding.size == 0:
                return None

            return embedding

        except (OSError, ValueError):
            return None

    @classmethod
    def _get_frame_dimensions(cls, frame_path, metadata_frame=None):
        width = None
        height = None

        if metadata_frame:
            width = cls._get_value(
                metadata_frame,
                "frame_width",
                "width"
            )
            height = cls._get_value(
                metadata_frame,
                "frame_height",
                "height"
            )

        if width and height:
            return float(width), float(height)

        image = cv2.imread(str(frame_path))

        if image is None:
            return None, None

        frame_height, frame_width = image.shape[:2]
        return float(frame_width), float(frame_height)

    @classmethod
    def _normalize_detection(cls, detection, frame_info, video_id):
        if not isinstance(detection, dict):
            return None

        category = cls._get_value(
            detection,
            "category",
            "class_name",
            "label",
            "name"
        )

        if not category:
            return None

        category = str(category).strip().lower()

        # Never allow people into the object tracker.
        if category == "person":
            return None

        bbox = cls._get_bbox(detection)

        if bbox is None:
            return None

        frame_stem = Path(frame_info["frame_filename"]).stem

        crop_path = cls._get_value(
            detection,
            "crop_path",
            "object_path",
            "image_path",
            "path"
        )

        if crop_path:
            crop_path = Path(crop_path)
            if not crop_path.is_absolute():
                crop_path = BASE_DIR / crop_path
        else:
            object_number = cls._get_value(
                detection,
                "object_number",
                "detection_number",
                "index",
                "id"
            )

            if object_number is None:
                return None

            crop_path = (
                UPLOAD_DIR
                / "video_objects"
                / str(video_id)
                / frame_stem
                / category
                / f"{category}_{object_number}.jpg"
            )

        # The feature extractor saves the same crop filename with .npy.
        embedding_path = (
            OBJECT_EMBEDDING_DIR
            / str(video_id)
            / frame_stem
            / category
            / f"{crop_path.stem}.npy"
        )

        embedding = cls._load_embedding(embedding_path)

        return {
            "category": category,
            "bbox": bbox,
            "crop_path": str(crop_path),
            "embedding_path": str(embedding_path),
            "embedding": embedding,
            "frame_number": frame_info["frame_number"],
            "source_frame_number": frame_info.get("source_frame_number"),
            "timestamp_seconds": frame_info.get("timestamp_seconds"),
            "frame_filename": frame_info["frame_filename"],
            "track_id": None
        }

    @classmethod
    def _association_score(cls, track, detection, frame_width, frame_height):
        # Category must match exactly.
        if track["category"] != detection["category"]:
            return None

        previous_box = track["last_bbox"]
        current_box = detection["bbox"]

        iou = cls._calculate_iou(previous_box, current_box)

        previous_center = cls._center(previous_box)
        current_center = cls._center(current_box)

        diagonal = np.sqrt(
            (frame_width ** 2) + (frame_height ** 2)
        )

        if diagonal <= 0:
            return None

        center_distance = np.sqrt(
            (previous_center[0] - current_center[0]) ** 2
            + (previous_center[1] - current_center[1]) ** 2
        )

        normalized_distance = center_distance / diagonal

        similarity = cls._cosine_similarity(
            track.get("last_embedding"),
            detection.get("embedding")
        )

        spatial_match = (
            iou >= cls.MIN_IOU_FOR_SPATIAL_MATCH
            or normalized_distance <= cls.MAX_NORMALIZED_DISTANCE
        )

        appearance_match = (
            similarity is not None
            and similarity >= cls.MIN_APPEARANCE_SIMILARITY
        )

        # Require at least one meaningful association signal.
        if not spatial_match and not appearance_match:
            return None

        spatial_score = max(
            iou,
            1.0 - min(normalized_distance, 1.0)
        )

        if similarity is None:
            return spatial_score

        return (0.65 * similarity) + (0.35 * spatial_score)

    @classmethod
    def _extract_frame_detections(cls, frame_info, video_id):
        detections = cls._get_value(
            frame_info,
            "objects",
            "detections",
            "detected_objects",
            default=[]
        )

        if not isinstance(detections, list):
            return []

        normalized = []

        for detection in detections:
            item = cls._normalize_detection(
                detection,
                frame_info,
                video_id
            )

            if item:
                normalized.append(item)

        return normalized

    @classmethod
    def track_objects(cls, video_id):
        video_id = str(video_id)

        detection_metadata_path = (
            OBJECT_DETECTION_DIR
            / video_id
            / "object_detection_metadata.json"
        )

        if not detection_metadata_path.exists():
            raise FileNotFoundError(
                f"Object detection metadata not found: "
                f"{detection_metadata_path}"
            )

        metadata = cls._read_json(detection_metadata_path)

        frames = metadata.get("frames", [])

        if not isinstance(frames, list) or not frames:
            raise ValueError(
                "No frame detection data found in object detection metadata."
            )

        frames = sorted(
            frames,
            key=lambda item: int(
                cls._get_value(item, "frame_number", default=0)
            )
        )

        tracks = {}
        next_track_number = 1
        processed_frames = []

        for frame_info in frames:
            frame_filename = cls._get_value(
                frame_info,
                "frame_filename",
                "filename"
            )

            if not frame_filename:
                frame_path_value = cls._get_value(
                    frame_info,
                    "frame_path",
                    "path"
                )
                if frame_path_value:
                    frame_filename = Path(frame_path_value).name

            if not frame_filename:
                continue

            frame_number = int(
                cls._get_value(
                    frame_info,
                    "frame_number",
                    "index",
                    default=0
                )
            )

            frame_info["frame_filename"] = frame_filename
            frame_info["frame_number"] = frame_number

            frame_path = FRAME_DIR / video_id / frame_filename

            if not frame_path.exists():
                supplied_path = cls._get_value(
                    frame_info,
                    "frame_path"
                )
                if supplied_path and Path(supplied_path).exists():
                    frame_path = Path(supplied_path)

            frame_width, frame_height = cls._get_frame_dimensions(
                frame_path,
                frame_info
            )

            if not frame_width or not frame_height:
                continue

            detections = cls._extract_frame_detections(
                frame_info,
                video_id
            )

            # Expire tracks that have been absent for too many frames.
            active_tracks = {
                track_id: track
                for track_id, track in tracks.items()
                if frame_number - track["last_frame_number"]
                <= cls.MAX_FRAME_GAP
            }

            candidates = []

            for track_id, track in active_tracks.items():
                for detection_index, detection in enumerate(detections):
                    score = cls._association_score(
                        track,
                        detection,
                        frame_width,
                        frame_height
                    )

                    if score is not None:
                        candidates.append(
                            (score, track_id, detection_index)
                        )

            # Greedy one-to-one matching: a track and a detection can
            # each be assigned only once in the current frame.
            candidates.sort(key=lambda item: item[0], reverse=True)

            matched_tracks = set()
            matched_detections = set()

            for score, track_id, detection_index in candidates:
                if track_id in matched_tracks:
                    continue

                if detection_index in matched_detections:
                    continue

                track = tracks[track_id]
                detection = detections[detection_index]

                detection["track_id"] = track_id

                track["last_bbox"] = detection["bbox"]
                track["last_frame_number"] = frame_number
                track["last_embedding"] = detection["embedding"]
                track["last_timestamp_seconds"] = detection[
                    "timestamp_seconds"
                ]

                track["detections"].append({
                    "frame_number": frame_number,
                    "timestamp_seconds": detection["timestamp_seconds"],
                    "frame_filename": frame_filename,
                    "bbox": detection["bbox"],
                    "crop_path": detection["crop_path"],
                    "embedding_path": detection["embedding_path"],
                    "association_score": round(float(score), 4)
                })

                matched_tracks.add(track_id)
                matched_detections.add(detection_index)

            # Unmatched detections begin new tracks.
            for detection_index, detection in enumerate(detections):
                if detection_index in matched_detections:
                    continue

                track_id = f"object_track_{next_track_number:05d}"
                next_track_number += 1

                detection["track_id"] = track_id

                tracks[track_id] = {
                    "track_id": track_id,
                    "category": detection["category"],
                    "first_frame_number": frame_number,
                    "last_frame_number": frame_number,
                    "first_timestamp_seconds": detection[
                        "timestamp_seconds"
                    ],
                    "last_timestamp_seconds": detection[
                        "timestamp_seconds"
                    ],
                    "last_bbox": detection["bbox"],
                    "last_embedding": detection["embedding"],
                    "detections": [{
                        "frame_number": frame_number,
                        "timestamp_seconds": detection["timestamp_seconds"],
                        "frame_filename": frame_filename,
                        "bbox": detection["bbox"],
                        "crop_path": detection["crop_path"],
                        "embedding_path": detection["embedding_path"],
                        "association_score": None
                    }]
                }

            processed_frames.append({
                "frame_number": frame_number,
                "frame_filename": frame_filename,
                "timestamp_seconds": frame_info.get(
                    "timestamp_seconds"
                ),
                "total_objects": len(detections),
                "objects": [
                    {
                        "category": item["category"],
                        "bbox": item["bbox"],
                        "track_id": item["track_id"],
                        "crop_path": item["crop_path"],
                        "embedding_found": (
                            item["embedding"] is not None
                        )
                    }
                    for item in detections
                ]
            })

        # Remove NumPy arrays from the final JSON representation.
        final_tracks = {}

        for track_id, track in tracks.items():
            final_tracks[track_id] = {
                "track_id": track_id,
                "category": track["category"],
                "first_frame_number": track["first_frame_number"],
                "last_frame_number": track["last_frame_number"],
                "first_timestamp_seconds": track[
                    "first_timestamp_seconds"
                ],
                "last_timestamp_seconds": track[
                    "last_timestamp_seconds"
                ],
                "total_detections": len(track["detections"]),
                "detections": track["detections"]
            }

        category_counts = {}

        for track in final_tracks.values():
            category = track["category"]
            category_counts[category] = (
                category_counts.get(category, 0) + 1
            )

        output_dir = OBJECT_TRACKING_DIR / video_id
        output_dir.mkdir(parents=True, exist_ok=True)

        output_path = output_dir / "object_tracking_metadata.json"

        result = {
            "video_id": video_id,
            "frames_processed": len(processed_frames),
            "total_object_detections": sum(
                frame["total_objects"] for frame in processed_frames
            ),
            "total_object_tracks": len(final_tracks),
            "category_track_counts": category_counts,
            "frames": processed_frames,
            "tracks": final_tracks,
            "metadata_path": str(output_path)
        }

        cls._write_json(output_path, result)

        return result
