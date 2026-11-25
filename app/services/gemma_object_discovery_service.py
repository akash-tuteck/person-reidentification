import base64
import json
import logging
import mimetypes
import re
from pathlib import Path

import requests


logger = logging.getLogger(__name__)


class GemmaObjectDiscoveryService:
    BASE_URL = "http://127.0.0.1:9931"
    CHAT_ENDPOINT = f"{BASE_URL}/v1/chat/completions"
    TIMEOUT = 240
    MAX_TOKENS = 800
    MAX_CATEGORIES = 20

    SYSTEM_PROMPT = """
You analyze CCTV images and discover meaningful, independently identifiable
physical objects that could be detected and tracked in separate video frames.

Return exactly one JSON object:
{"objects":[{"category":"short generic object category","description":"brief visible description"}]}

Rules:
- Include visible objects such as vehicles, bags, furniture, animals and other
  meaningful physical objects.
- Exclude people, body parts, clothing worn by people, object parts, tiny details,
  text, shadows, reflections and background textures.
- Do not invent objects. Include only objects clearly visible in the image.
- Use short generic category names (for example: car, backpack, dog, chair).
- Include each category once. Return at most 20 categories.
- If no qualifying objects are visible, return {"objects":[]}.
- JSON only; no markdown or explanation.
"""

    @classmethod
    def discover_objects(cls, image_path: str | Path) -> dict:
        path = Path(image_path)
        if not path.is_file():
            raise FileNotFoundError(f"Image not found: {path}")

        mime_type, _ = mimetypes.guess_type(str(path))
        if not mime_type or not mime_type.startswith("image/"):
            mime_type = "image/jpeg"

        image_data = base64.b64encode(path.read_bytes()).decode("utf-8")
        image_url = f"data:{mime_type};base64,{image_data}"

        payload = {
            "messages": [
                {"role": "system", "content": cls.SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                "Identify all clearly visible meaningful objects "
                                "that can be separately detected and tracked. "
                                "Return the required JSON."
                            ),
                        },
                        {"type": "image_url", "image_url": {"url": image_url}},
                    ],
                },
            ],
            "temperature": 0.1,
            "max_tokens": cls.MAX_TOKENS,
            "stream": False,
            "response_format": {"type": "json_object"},
        }

        try:
            response = requests.post(
                cls.CHAT_ENDPOINT, json=payload, timeout=cls.TIMEOUT
            )
            response.raise_for_status()
            response_data = response.json()
            choices = response_data.get("choices", [])
            if not choices:
                raise ValueError("Gemma API returned no choices")

            message = choices[0].get("message", {})
            content = message.get("content", "")
            if isinstance(content, list):
                content = "\n".join(
                    item.get("text", "")
                    for item in content
                    if isinstance(item, dict)
                )
            if not isinstance(content, str) or not content.strip():
                raise ValueError("Gemma returned empty content")

            content = content.strip()
            content = re.sub(
                r"^\s*```(?:json)?\s*|\s*```\s*$",
                "",
                content,
                flags=re.IGNORECASE,
            )
            result = json.loads(content)
            raw_objects = result.get("objects", [])
            if not isinstance(raw_objects, list):
                raise ValueError("Gemma objects field must be a list")

            objects = []
            seen = set()
            for item in raw_objects:
                if not isinstance(item, dict):
                    continue
                category = item.get("category")
                description = item.get("description", "")
                if not isinstance(category, str):
                    continue
                category = re.sub(r"[^a-z0-9 -]", "", category.lower()).strip()
                category = " ".join(category.split())
                if not category or category in seen:
                    continue
                if not isinstance(description, str):
                    description = ""
                objects.append({
                    "category": category[:64],
                    "description": description.strip()[:300],
                })
                seen.add(category)
                if len(objects) >= cls.MAX_CATEGORIES:
                    break

            return {
                "success": True,
                "model": "Gemma 4",
                "objects": objects,
                "total_categories": len(objects),
            }

        except requests.exceptions.RequestException as exc:
            logger.exception("Gemma object discovery request failed")
            raise RuntimeError(
                "Cannot reach local Gemma server; check llama-server."
            ) from exc
        except (KeyError, TypeError, json.JSONDecodeError, ValueError) as exc:
            logger.exception("Invalid Gemma object discovery response")
            raise RuntimeError(f"Gemma discovery failed: {exc}") from exc
