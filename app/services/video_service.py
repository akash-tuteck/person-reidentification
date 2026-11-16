
import cv2

from pathlib import Path
from uuid import uuid4


BASE_DIR = Path(__file__).resolve().parent.parent

UPLOAD_DIR = BASE_DIR / "uploads"

VIDEO_DIR = UPLOAD_DIR / "videos"
FRAME_DIR = UPLOAD_DIR / "video_frames"

SUPPORTED_EXTENSIONS = {
    ".mp4",
    ".avi",
    ".mov",
    ".mkv"
}


class VideoProcessingService:

    @staticmethod
    def process_video(
        file_bytes: bytes,
        filename: str,
        frame_interval_seconds: float = 1.0
    ) -> dict:

        if not file_bytes:
            raise ValueError("Uploaded video is empty")

        if frame_interval_seconds <= 0:
            raise ValueError(
                "Frame interval must be greater than zero"
            )

        extension = Path(filename).suffix.lower()

        if extension not in SUPPORTED_EXTENSIONS:
            raise ValueError(
                f"Unsupported video format: {extension}"
            )

        # Create existing storage directories if necessary
        VIDEO_DIR.mkdir(parents=True, exist_ok=True)
        FRAME_DIR.mkdir(parents=True, exist_ok=True)

        video_id = str(uuid4())

        video_path = VIDEO_DIR / f"{video_id}{extension}"
        frames_directory = FRAME_DIR / video_id

        frames_directory.mkdir(
            parents=True,
            exist_ok=True
        )

        # Save original video
        video_path.write_bytes(file_bytes)

        capture = cv2.VideoCapture(str(video_path))

        if not capture.isOpened():
            capture.release()
            video_path.unlink(missing_ok=True)
            frames_directory.rmdir()

            raise ValueError(
                "Unable to open uploaded video"
            )

        fps = capture.get(cv2.CAP_PROP_FPS)

        total_frames = int(
            capture.get(cv2.CAP_PROP_FRAME_COUNT)
        )

        if fps <= 0 or total_frames <= 0:
            capture.release()
            video_path.unlink(missing_ok=True)
            frames_directory.rmdir()

            raise ValueError(
                "Invalid video metadata"
            )

        duration_seconds = total_frames / fps

        # Calculate frame extraction interval
        frame_interval = max(
            1,
            round(fps * frame_interval_seconds)
        )

        extracted_frames = []
        frame_index = 0

        try:

            while True:

                success, frame = capture.read()

                if not success:
                    break

                if frame_index % frame_interval == 0:

                    timestamp_ms = round(
                        (frame_index / fps) * 1000
                    )

                    frame_number = len(extracted_frames) + 1

                    frame_filename = (
                        f"frame_{frame_number:06d}_"
                        f"at_{timestamp_ms}ms.jpg"
                    )

                    frame_path = (
                        frames_directory / frame_filename
                    )

                    saved = cv2.imwrite(
                        str(frame_path),
                        frame
                    )

                    if not saved:
                        raise RuntimeError(
                            "Failed to save extracted frame"
                        )

                    extracted_frames.append({
                        "frame_number": frame_number,
                        "source_frame_index": frame_index,
                        "timestamp_ms": timestamp_ms,
                        "frame_filename": frame_filename,
                        "frame_path": str(frame_path)
                    })

                frame_index += 1

            if not extracted_frames:
                raise ValueError(
                    "No frames could be extracted from video"
                )

        except Exception:

            for frame_file in frames_directory.glob("*.jpg"):
                frame_file.unlink(missing_ok=True)

            frames_directory.rmdir()
            video_path.unlink(missing_ok=True)

            raise

        finally:
            capture.release()

        return {
            "video_id": video_id,
            "original_filename": filename,
            "video_path": str(video_path),
            "fps": round(fps, 2),
            "total_video_frames": total_frames,
            "duration_seconds": round(duration_seconds, 2),
            "frame_interval_seconds": frame_interval_seconds,
            "total_extracted_frames": len(extracted_frames),
            "frames_directory": str(frames_directory),
            "extracted_frames": extracted_frames,
            "status": "completed"
        }

    @staticmethod
    def get_video_path(video_id: str):

        if not video_id or Path(video_id).name != video_id:
            return None

        for extension in SUPPORTED_EXTENSIONS:

            video_path = VIDEO_DIR / f"{video_id}{extension}"

            if video_path.exists():
                return video_path

        return None

    @staticmethod
    def get_frames_directory(video_id: str):

        if not video_id or Path(video_id).name != video_id:
            return None

        frames_directory = FRAME_DIR / video_id

        if frames_directory.exists():
            return frames_directory

        return None
