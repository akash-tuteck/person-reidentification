
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from app.services.grounding_dino_service import GroundingDINOService


IMAGE_ID = "90af2f07-c48e-4132-8b19-8cb125658acf"

result = GroundingDINOService.detect_objects(
    image_id=IMAGE_ID,
    prompts=[
        "person",
        "backpack",
        "handbag",
        "chair",
        "bottle",
        "laptop",
        'car',
        'bird'
    ],
    threshold=0.35,
    text_threshold=0.25
)

print("\nGROUNDING DINO TEST")
print("===================")

print("Image ID:", result["image_id"])
print("Model:", result["model"])
print("Device:", result["device"])

print("Total objects:", result["total_objects"])
print("Category counts:", result["category_counts"])

print("\nDETECTIONS")
print("==========")

for obj in result["objects"]:

    print(
        obj["label"],
        "| Confidence:", obj["confidence"],
        "| Box:", obj["bounding_box"],
        "| Crop:", obj["crop_filename"]
    )

print("\nAnnotated image:", result["annotated_image_path"])
print("Metadata:", result["metadata_path"])
