
from pathlib import Path
import json
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.services.gemma_service import GemmaService


IMAGE_ID = "d0e0c0f5-4c0c-4a66-9627-afbf15461169"

IMAGE_PATH = (
    Path("app/uploads/processed")
    / f"{IMAGE_ID}.jpg"
)

if not IMAGE_PATH.exists():
    raise FileNotFoundError(
        f"Test image not found: {IMAGE_PATH}"
    )

print("Testing Gemma object discovery...")
print("Image:", IMAGE_PATH)

result = GemmaService.discover_objects(IMAGE_PATH)

print("\nDISCOVERY RESULT")
print(json.dumps(result, indent=4))

print("\nSUMMARY")
print("Success:", result["success"])
print("Total categories:", result["total_categories"])

for item in result["objects"]:
    print(
        f"- {item['category']}: {item['description']}"
    )
