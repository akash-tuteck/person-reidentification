
import base64
import json
import logging
import mimetypes

import requests

from pathlib import Path
from typing import Any, Optional


logger = logging.getLogger(__name__)


class GemmaService:

    BASE_URL = "http://127.0.0.1:9931"
    CHAT_ENDPOINT = f"{BASE_URL}/v1/chat/completions"

    TIMEOUT = 240
    MAX_TOKENS = 800

    SYSTEM_PROMPT = """
You are an object description assistant for a CCTV object analysis system.

Analyze the supplied image and describe only characteristics that are
visibly supported by the image.

Return exactly one valid JSON object with these fields:

{
    "category": "string",
    "description": "string",
    "color": "string or null",
    "shape": "string or null",
    "surface_or_material": "string or null",
    "condition": "string or null",
    "visible_characteristics": ["string"]
}

Rules:
1. Describe the main visible object.
2. Do not invent details that cannot be seen.
3. Do not make medical or health assessments.
4. Do not identify people.
5. Do not infer ownership, identity, or intent.
6. Use null when an attribute cannot be determined.
7. Keep descriptions concise and factual.
8. Return JSON only.
9. Do not include Markdown, explanations, or thinking text.
"""

    @classmethod
    def _encode_image(cls, image_path: str | Path) -> tuple[str, str]:

        image_path = Path(image_path)

        if not image_path.exists():
            raise FileNotFoundError(
                f"Image not found: {image_path}"
            )

        if not image_path.is_file():
            raise ValueError(
                f"Path is not a file: {image_path}"
            )

        mime_type, _ = mimetypes.guess_type(str(image_path))

        if not mime_type or not mime_type.startswith("image/"):
            mime_type = "image/jpeg"

        with image_path.open("rb") as image_file:
            image_data = base64.b64encode(
                image_file.read()
            ).decode("utf-8")

        return image_data, mime_type

    @classmethod
    def _extract_message_content(cls, message: dict) -> str:

        content = message.get("content")

        # Standard OpenAI-compatible string response.
        if isinstance(content, str):
            return content.strip()

        # Some servers return content as a list of text blocks.
        if isinstance(content, list):

            text_parts = []

            for item in content:

                if isinstance(item, dict):
                    text = item.get("text")

                    if isinstance(text, str):
                        text_parts.append(text)

            return "\n".join(text_parts).strip()

        return ""

    @classmethod
    def _extract_json(cls, response_text: str) -> dict[str, Any]:

        if not isinstance(response_text, str):
            raise ValueError(
                "Gemma response is not a string"
            )

        response_text = response_text.strip()

        if not response_text:
            raise ValueError(
                "Gemma returned an empty content response"
            )

        # Remove Markdown code fences.
        if response_text.startswith("```"):

            lines = response_text.splitlines()

            if lines and lines[0].startswith("```"):
                lines = lines[1:]

            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]

            response_text = "\n".join(lines).strip()

        # First try parsing the complete response.
        try:

            result = json.loads(response_text)

        except json.JSONDecodeError:

            # Recover a JSON object surrounded by extra text.
            start = response_text.find("{")

            if start == -1:
                raise ValueError(
                    "No JSON object found in Gemma response"
                )

            # Find the matching closing brace while respecting
            # braces inside quoted JSON strings.
            depth = 0
            in_string = False
            escaped = False
            end = -1

            for index in range(start, len(response_text)):

                char = response_text[index]

                if in_string:

                    if escaped:
                        escaped = False

                    elif char == "\\":
                        escaped = True

                    elif char == '"':
                        in_string = False

                    continue

                if char == '"':
                    in_string = True

                elif char == "{":
                    depth += 1

                elif char == "}":
                    depth -= 1

                    if depth == 0:
                        end = index
                        break

            if end == -1:
                raise ValueError(
                    "Gemma returned incomplete or truncated JSON"
                )

            try:

                result = json.loads(
                    response_text[start:end + 1]
                )

            except json.JSONDecodeError as exc:

                raise ValueError(
                    "Gemma returned malformed JSON"
                ) from exc

        if not isinstance(result, dict):
            raise ValueError(
                "Gemma response must be a JSON object"
            )

        return result

    @classmethod
    def _normalize_response(
        cls,
        result: dict[str, Any],
        detected_category: Optional[str] = None
    ) -> dict[str, Any]:

        # Detector category remains authoritative.
        category = detected_category or result.get("category")

        if not isinstance(category, str) or not category.strip():
            raise ValueError(
                "Gemma response does not contain a valid category"
            )

        description = result.get("description")

        if not isinstance(description, str) or not description.strip():
            raise ValueError(
                "Gemma response does not contain a valid description"
            )

        def normalize_optional_string(value):

            if not isinstance(value, str):
                return None

            value = value.strip()

            return value if value else None

        visible_characteristics = result.get(
            "visible_characteristics", []
        )

        if not isinstance(visible_characteristics, list):
            visible_characteristics = []

        visible_characteristics = [
            item.strip()
            for item in visible_characteristics
            if isinstance(item, str) and item.strip()
        ]

        return {
            "category": category.strip().lower(),
            "description": description.strip(),
            "color": normalize_optional_string(
                result.get("color")
            ),
            "shape": normalize_optional_string(
                result.get("shape")
            ),
            "surface_or_material": normalize_optional_string(
                result.get("surface_or_material")
                or result.get("material")
            ),
            "condition": normalize_optional_string(
                result.get("condition")
            ),
            "visible_characteristics": visible_characteristics
        }

    @classmethod
    def describe_object(
        cls,
        image_path: str | Path,
        detected_category: Optional[str] = None
    ) -> dict[str, Any]:

        image_data, mime_type = cls._encode_image(image_path)

        image_url = (
            f"data:{mime_type};base64,{image_data}"
        )

        user_prompt = (
            "Examine the supplied image. "
            "Return one complete JSON object following the system schema. "
            "Do not provide reasoning or any text outside the JSON."
        )

        if detected_category:

            user_prompt += (
                f" The detector category is '{detected_category}'. "
                "Preserve this category. Describe only visible evidence."
            )

        payload = {
            "messages": [
                {
                    "role": "system",
                    "content": cls.SYSTEM_PROMPT
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": user_prompt
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": image_url
                            }
                        }
                    ]
                }
            ],
            "temperature": 0.1,
            "max_tokens": cls.MAX_TOKENS,
            "stream": False,

            # Ask llama-server for JSON-constrained output.
            "response_format": {
                "type": "json_object"
            }
        }

        try:

            response = requests.post(
                cls.CHAT_ENDPOINT,
                json=payload,
                timeout=cls.TIMEOUT
            )

            response.raise_for_status()

            response_data = response.json()

            choices = response_data.get("choices", [])

            if not choices:
                raise ValueError(
                    "Gemma API returned no choices"
                )

            choice = choices[0]

            message = choice.get("message", {})

            if not isinstance(message, dict):
                raise ValueError(
                    "Gemma returned an invalid message structure"
                )

            content = cls._extract_message_content(message)

            if not content:

                finish_reason = choice.get("finish_reason")

                reasoning = message.get(
                    "reasoning_content",
                    message.get("reasoning", "")
                )

                reasoning_length = (
                    len(reasoning)
                    if isinstance(reasoning, str)
                    else 0
                )

                logger.error(
                    "Empty Gemma response. finish_reason=%s, "
                    "reasoning_length=%s, usage=%s",
                    finish_reason,
                    reasoning_length,
                    response_data.get("usage")
                )

                raise ValueError(
                    "Gemma returned empty content. "
                    f"finish_reason={finish_reason}, "
                    f"reasoning_length={reasoning_length}. "
                    "Check llama-server generation settings."
                )

            logger.info(
                "Gemma response received. finish_reason=%s, "
                "content_length=%s",
                choice.get("finish_reason"),
                len(content)
            )

            parsed_result = cls._extract_json(content)

            normalized_result = cls._normalize_response(
                parsed_result,
                detected_category
            )

            return {
                "success": True,
                "model": "Gemma 4",
                "data": normalized_result
            }

        except requests.exceptions.ConnectionError as exc:

            logger.exception(
                "Cannot connect to local Gemma server"
            )

            raise RuntimeError(
                "Local Gemma server is unavailable. "
                "Check whether llama-server is running."
            ) from exc

        except requests.exceptions.Timeout as exc:

            logger.exception(
                "Gemma inference timed out"
            )

            raise RuntimeError(
                "Gemma inference timed out."
            ) from exc

        except requests.exceptions.RequestException as exc:

            logger.exception(
                "Gemma API request failed"
            )

            raise RuntimeError(
                f"Gemma API request failed: {exc}"
            ) from exc

        except (KeyError, IndexError, TypeError) as exc:

            logger.exception(
                "Unexpected Gemma API response"
            )

            raise RuntimeError(
                "Unexpected response format from Gemma."
            ) from exc

        except ValueError as exc:

            logger.exception(
                "Gemma returned invalid structured data"
            )

            raise RuntimeError(
                f"Invalid Gemma response: {exc}"
            ) from exc
