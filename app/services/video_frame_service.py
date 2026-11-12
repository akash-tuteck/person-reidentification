
import cv2

from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent

VIDEO_DIR = BASE_DIR / "uploads" / "videos"
FRAME_DIR = BASE_DIR / "uploads" / "frames"


class VideoFrameService:

    @staticmethod
    def extract_frames(video_id, interval_seconds=1):

        if interval_seconds <= 0:
            raise ValueError(
                "Interval must be greater than zero"
            )

        video_path = VIDEO_DIR / f"{video_id}.mp4"

        if not video_path.exists():
            raise FileNotFoundError(
                f"Video not found: {video_path}"
            )

        output_dir = FRAME_DIR / str(video_id)
        output_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        capture = cv2.VideoCapture(str(video_path))

        if not capture.isOpened():
            raise ValueError(
                "Unable to open video"
            )

        fps = capture.get(cv2.CAP_PROP_FPS)

        total_frames = int(
            capture.get(cv2.CAP_PROP_FRAME_COUNT)
        )

        if fps <= 0 or total_frames <= 0:
            capture.release()

            raise ValueError(
                "Invalid video metadata"
            )

        duration = total_frames / fps

        frame_interval = max(
            1,
            round(fps * interval_seconds)
        )

        extracted_frames = []

        frame_number = 0
        saved_count = 0

        try:

            while True:

                success, frame = capture.read()

                if not success:
                    break

                if frame_number % frame_interval == 0:

                    timestamp = frame_number / fps

                    frame_name = (
                        f"frame_{saved_count + 1:06d}.jpg"
                    )

                    frame_path = output_dir / frame_name

                    saved = cv2.imwrite(
                        str(frame_path),
                        frame
                    )

                    if not saved:
                        raise IOError(
                            f"Unable to save frame: {frame_name}"
                        )

                    extracted_frames.append({

                        "frame_number": saved_count + 1,

                        "source_frame_number": frame_number,

                        "timestamp_seconds": round(
                            timestamp,
                            3
                        ),

                        "frame_path": str(frame_path)

                    })

                    saved_count += 1

                frame_number += 1

        finally:

            capture.release()

        return {

            "video_id": str(video_id),

            "fps": round(fps, 2),

            "total_video_frames": total_frames,

            "duration_seconds": round(
                duration,
                2
            ),

            "interval_seconds": interval_seconds,

            "total_extracted_frames": saved_count,

            "frame_folder": str(output_dir),

            "frames": extracted_frames

        }
