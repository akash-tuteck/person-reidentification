import json
import logging
import re
import shutil
from pathlib import Path

import torch
from PIL import Image, ImageDraw, ImageFont
from transformers import (
    AutoProcessor,
    AutoModelForZeroShotObjectDetection,
)

from app.services.image_service import ImageProcessingService
from app.services.gemma_object_discovery_service import (
    GemmaObjectDiscoveryService,
)


logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent
UPLOAD_DIR = BASE_DIR / "uploads"

GROUNDING_DIR = UPLOAD_DIR / "grounding_dino"
DETECTION_DIR = GROUNDING_DIR / "detections"
OBJECT_DIR = GROUNDING_DIR / "objects"
METADATA_DIR = GROUNDING_DIR / "metadata"

for directory in (DETECTION_DIR, OBJECT_DIR, METADATA_DIR):
    directory.mkdir(parents=True, exist_ok=True)


class GroundingDINOService:
    MODEL_NAME = "IDEA-Research/grounding-dino-tiny"
    DEFAULT_THRESHOLD = 0.35
    DEFAULT_TEXT_THRESHOLD = 0.25
    MAX_PROMPTS = 20
    MAX_DETECTIONS = 300

    _model = None
    _processor = None
    _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @classmethod
    def get_model(cls):
        if cls._model is None:
            logger.info("Loading Grounding DINO model: %s", cls.MODEL_NAME)
            cls._processor = AutoProcessor.from_pretrained(cls.MODEL_NAME)
            cls._model = AutoModelForZeroShotObjectDetection.from_pretrained(
                cls.MODEL_NAME
            )
            cls._model.to(cls._device)
            cls._model.eval()
            logger.info("Grounding DINO loaded on %s", cls._device)
        return cls._model

    @staticmethod
    def _validate_prompts(prompts: list[str]) -> list[str]:
        if not isinstance(prompts, list) or not prompts:
            raise ValueError("At least one detection prompt is required")
        if len(prompts) > GroundingDINOService.MAX_PROMPTS:
            raise ValueError("Maximum 20 detection prompts are allowed")

        normalized = []
        for prompt in prompts:
            if not isinstance(prompt, str):
                raise ValueError("Every detection prompt must be a string")
            prompt = prompt.strip().lower()
            if not prompt:
                raise ValueError("Detection prompts cannot be empty")
            if len(prompt) > 100:
                raise ValueError(
                    "Each detection prompt must be at most 100 characters"
                )
            prompt = " ".join(prompt.replace(".", " ").split())
            if prompt:
                normalized.append(prompt)

        normalized = list(dict.fromkeys(normalized))
        if not normalized:
            raise ValueError("No valid detection prompts supplied")
        return normalized

    @staticmethod
    def _safe_category_name(label: str) -> str:
        category = label.lower().strip()
        category = re.sub(r"[^a-z0-9_-]+", "_", category)
        category = category.strip("_") or "unknown_object"
        return category[:64]

    @classmethod
    def detect_objects_automatically(
        cls,
        image_id: str,
        threshold: float = DEFAULT_THRESHOLD,
        text_threshold: float = DEFAULT_TEXT_THRESHOLD,
    ) -> dict:
        image_path = ImageProcessingService.get_processed_image_path(image_id)
        if image_path is None:
            raise ValueError(f"Processed image not found: {image_id}")

        discovery = GemmaObjectDiscoveryService.discover_objects(image_path)
        prompts = [
            item["category"]
            for item in discovery["objects"]
            if isinstance(item, dict)
            and isinstance(item.get("category"), str)
            and item["category"].strip()
        ]
        prompts = list(dict.fromkeys(prompt.strip().lower() for prompt in prompts))
        if not prompts:
            return {
                "success": True,
                "image_id": image_id,
                "message": "Gemma did not discover any qualifying objects",
                "discovery": discovery,
                "detection": None,
            }

        detection = cls.detect_objects(
            image_id=image_id,
            prompts=prompts[:cls.MAX_PROMPTS],
            threshold=threshold,
            text_threshold=text_threshold,
        )
        return {
            "success": True,
            "image_id": image_id,
            "discovery": discovery,
            "detection": detection,
        }

    @classmethod
    def detect_objects(
        cls,
        image_id: str,
        prompts: list[str],
        threshold: float = DEFAULT_THRESHOLD,
        text_threshold: float = DEFAULT_TEXT_THRESHOLD,
    ) -> dict:
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("Detection threshold must be between 0 and 1")
        if not 0.0 <= text_threshold <= 1.0:
            raise ValueError("Text threshold must be between 0 and 1")

        prompts = cls._validate_prompts(prompts)
        image_path = ImageProcessingService.get_processed_image_path(image_id)
        if image_path is None:
            raise ValueError(f"Processed image not found: {image_id}")

        with Image.open(image_path) as source_image:
            image = source_image.convert("RGB")
        width, height = image.size

        image_object_dir = OBJECT_DIR / image_id
        image_metadata_dir = METADATA_DIR / image_id
        for directory in (image_object_dir, image_metadata_dir):
            if directory.exists():
                shutil.rmtree(directory)
            directory.mkdir(parents=True, exist_ok=True)

        model = cls.get_model()
        text_prompt = ". ".join(prompts) + "."
        inputs = cls._processor(
            images=image, text=text_prompt, return_tensors="pt"
        )
        inputs = {key: value.to(cls._device) for key, value in inputs.items()}

        with torch.inference_mode():
            outputs = model(**inputs)
            results = cls._processor.post_process_grounded_object_detection(
                outputs,
                input_ids=inputs["input_ids"],
                threshold=threshold,
                text_threshold=text_threshold,
                target_sizes=[(height, width)],
            )

        result = results[0]
        boxes = result["boxes"].cpu().tolist()
        scores = result["scores"].cpu().tolist()
        labels = result.get("text_labels", result.get("labels", []))

        if len(boxes) > cls.MAX_DETECTIONS:
            boxes = boxes[:cls.MAX_DETECTIONS]
            scores = scores[:cls.MAX_DETECTIONS]
            labels = labels[:cls.MAX_DETECTIONS]

        annotated_image = image.copy()
        draw = ImageDraw.Draw(annotated_image)
        font = ImageFont.load_default()
        detected_objects = []
        category_counts = {}

        for box, score, label in zip(boxes, scores, labels):
            label = str(label).strip()
            if not label:
                continue

            x1, y1, x2, y2 = map(int, map(round, box))
            x1 = max(0, min(x1, width))
            y1 = max(0, min(y1, height))
            x2 = max(0, min(x2, width))
            y2 = max(0, min(y2, height))
            if x2 <= x1 or y2 <= y1:
                continue

            category = cls._safe_category_name(label)
            category_dir = image_object_dir / category
            category_dir.mkdir(parents=True, exist_ok=True)
            category_counts[category] = category_counts.get(category, 0) + 1
            object_number = category_counts[category]
            crop_filename = f"object_{object_number:03d}.jpg"
            crop_path = category_dir / crop_filename

            image.crop((x1, y1, x2, y2)).save(
                crop_path, format="JPEG", quality=95
            )
            confidence = float(score)
            draw.rectangle((x1, y1, x2, y2), outline="red", width=3)
            draw.text(
                (x1, max(0, y1 - 15)),
                f"{label} {confidence:.2f}",
                fill="red",
                font=font,
            )
            detected_objects.append({
                "object_number": object_number,
                "category": category,
                "label": label,
                "confidence": round(confidence, 4),
                "bounding_box": {
                    "x1": x1, "y1": y1, "x2": x2, "y2": y2
                },
                "crop_filename": crop_filename,
                "crop_path": str(crop_path),
            })

        annotated_path = DETECTION_DIR / f"{image_id}.jpg"
        annotated_image.save(annotated_path, format="JPEG", quality=95)

        metadata = {
            "image_id": image_id,
            "model": cls.MODEL_NAME,
            "device": str(cls._device),
            "prompts": prompts,
            "threshold": threshold,
            "text_threshold": text_threshold,
            "image_width": width,
            "image_height": height,
            "total_objects": len(detected_objects),
            "category_counts": category_counts,
            "objects": detected_objects,
        }
        metadata_path = image_metadata_dir / "detection_metadata.json"
        with metadata_path.open("w", encoding="utf-8") as file:
            json.dump(metadata, file, indent=4, ensure_ascii=False)

        return {
            "image_id": image_id,
            "model": cls.MODEL_NAME,
            "device": str(cls._device),
            "prompts": prompts,
            "total_objects": len(detected_objects),
            "category_counts": category_counts,
            "annotated_image_path": str(annotated_path),
            "metadata_path": str(metadata_path),
            "objects": detected_objects,
        }
