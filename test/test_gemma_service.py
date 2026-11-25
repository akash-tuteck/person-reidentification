import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.services.gemma_service import GemmaService  # now it can find `app`

IMAGE_PATH = PROJECT_ROOT / "app/uploads/objects/3f55c3b2-3493-4d4d-9d22-3cf08cc4538a/dog/dog_1.jpg"


if __name__ == "__main__":

    print("\nGEMMA OBJECT DESCRIPTION TEST")
    print("=============================")

    result = GemmaService.describe_object(
        image_path=IMAGE_PATH,
        detected_category="dog"
    )

    print("\nRESULT")
    print("=============================")

    print(result)
