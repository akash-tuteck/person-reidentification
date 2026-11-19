
import json
from pathlib import Path

import cv2


BASE_DIR = Path(__file__).resolve().parent.parent
UPLOAD_DIR = BASE_DIR / "uploads"

FRAME_DIR = UPLOAD_DIR / "frames"
PERSON_TRACKING_DIR = UPLOAD_DIR / "video_tracking"
OBJECT_TRACKING_DIR = UPLOAD_DIR / "video_object_tracking"
ANNOTATED_VIDEO_DIR = UPLOAD_DIR / "annotated_videos"


class VideoAnnotationService:

    # OpenCV uses BGR color format.
    PERSON_COLOR = (0, 255, 0)
    OBJECT_COLOR = (255, 165, 0)

    @staticmethod
    def _read_json(path):
        with open(path, "r", encoding="utf-8") as file:
            return json.load(file)

    @staticmethod
    def _draw_label(frame, text, x, y, color):
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.55
        thickness = 2

        (text_width, text_height), baseline = cv2.getTextSize(
            text,
            font,
            font_scale,
            thickness
        )

        x = max(0, int(x))
        y = max(text_height + baseline, int(y))

        cv2.rectangle(
            frame,
            (x, y - text_height - baseline - 5),
            (x + text_width + 8, y),
            color,
            -1
        )

        cv2.putText(
            frame,
            text,
            (x + 4, y - baseline - 2),
            font,
            font_scale,
            (0, 0, 0),
            thickness,
            cv2.LINE_AA
        )

    @classmethod
    def _draw_box(cls, frame, bbox, label, color):
        if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
            return

        x1, y1, x2, y2 = map(int, bbox)

        height, width = frame.shape[:2]

        x1 = max(0, min(x1, width - 1))
        y1 = max(0, min(y1, height - 1))
        x2 = max(0, min(x2, width - 1))
        y2 = max(0, min(y2, height - 1))

        if x2 <= x1 or y2 <= y1:
            return

        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            color,
            2
        )

        cls._draw_label(
            frame,
            label,
            x1,
            y1,
            color
        )

    @classmethod
    def _annotate_persons(cls, frame, frame_data):
        persons = frame_data.get("persons", [])

        for person in persons:
            track_id = person.get("track_id")
            bbox = person.get("bbox")

            if not track_id or bbox is None:
                continue

            label = f"PERSON | {track_id}"

            cls._draw_box(
                frame,
                bbox,
                label,
                cls.PERSON_COLOR
            )

    @classmethod
    def _annotate_objects(cls, frame, frame_data):
        objects = frame_data.get("objects", [])

        for obj in objects:
            track_id = obj.get("track_id")
            category = obj.get("category")
            bbox = obj.get("bbox")

            if not track_id or not category or bbox is None:
                continue

            label = f"{category.upper()} | {track_id}"

            cls._draw_box(
                frame,
                bbox,
                label,
                cls.OBJECT_COLOR
            )

    @classmethod
    def generate_annotated_video(cls, video_id, output_fps=1.0):
        video_id = str(video_id)

        if output_fps <= 0:
            raise ValueError("Output FPS must be greater than zero.")

        person_metadata_path = (
            PERSON_TRACKING_DIR
            / video_id
            / "tracking_metadata.json"
        )

        object_metadata_path = (
            OBJECT_TRACKING_DIR
            / video_id
            / "object_tracking_metadata.json"
        )

        if not person_metadata_path.exists():
            raise FileNotFoundError(
                f"Person tracking metadata not found: "
                f"{person_metadata_path}"
            )

        if not object_metadata_path.exists():
            raise FileNotFoundError(
                f"Object tracking metadata not found: "
                f"{object_metadata_path}"
            )

        person_metadata = cls._read_json(person_metadata_path)
        object_metadata = cls._read_json(object_metadata_path)

        person_frames = {
            item["frame_filename"]: item
            for item in person_metadata.get("frames", [])
            if item.get("frame_filename")
        }

        object_frames = {
            item["frame_filename"]: item
            for item in object_metadata.get("frames", [])
            if item.get("frame_filename")
        }

        frame_filenames = sorted(
            set(person_frames.keys()) | set(object_frames.keys())
        )

        if not frame_filenames:
            raise ValueError("No tracked frames were found.")

        output_dir = ANNOTATED_VIDEO_DIR / video_id
        output_dir.mkdir(parents=True, exist_ok=True)

        output_path = output_dir / "annotated_video.mp4"

        writer = None
        processed_frames = 0

        try:
            for filename in frame_filenames:
                frame_path = FRAME_DIR / video_id / filename

                if not frame_path.exists():
                    raise FileNotFoundError(
                        f"Extracted frame not found: {frame_path}"
                    )

                frame = cv2.imread(str(frame_path))

                if frame is None:
                    raise ValueError(
                        f"Could not read frame: {frame_path}"
                    )

                if writer is None:
                    height, width = frame.shape[:2]

                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

                    writer = cv2.VideoWriter(
                        str(output_path),
                        fourcc,
                        float(output_fps),
                        (width, height)
                    )

                    if not writer.isOpened():
                        raise RuntimeError(
                            "Could not initialize MP4 video writer."
                        )

                person_data = person_frames.get(filename, {})
                object_data = object_frames.get(filename, {})

                cls._annotate_persons(frame, person_data)
                cls._annotate_objects(frame, object_data)

                timestamp = person_data.get(
                    "timestamp_seconds",
                    object_data.get("timestamp_seconds", 0)
                )

                cv2.putText(
                    frame,
                    f"Time: {timestamp:.1f}s",
                    (15, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA
                )

                writer.write(frame)
                processed_frames += 1

        finally:
            if writer is not None:
                writer.release()

        if not output_path.exists() or output_path.stat().st_size == 0:
            raise RuntimeError("Annotated video generation failed.")

        return {
            "video_id": video_id,
            "frames_annotated": processed_frames,
            "output_fps": float(output_fps),
            "total_person_tracks": person_metadata.get(
                "total_tracks_created", 0
            ),
            "total_object_tracks": object_metadata.get(
                "total_object_tracks", 0
            ),
            "output_path": str(output_path),
            "file_size_bytes": output_path.stat().st_size
        }
