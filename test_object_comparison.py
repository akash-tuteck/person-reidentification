
from app.controllers.image_controller import ImageController

original_image_id = "95ce8ac8-d586-495c-bd86-48a9c9389f33"

search_image_ids = [
    "d0e0c0f5-4c0c-4a66-9627-afbf15461169",
    "3f55c3b2-3493-4d4d-9d22-3cf08cc4538a"
]

result = ImageController.compare_objects(
    original_image_id=original_image_id,
    search_image_ids=search_image_ids,
    top_k=5
)

print("\nOBJECT COMPARISON RESULTS")
print("=========================")

print("Original image:", result["original_image_id"])
print("Total objects:", result["total_original_objects"])

for item in result["results"]:

    print("\nOriginal object:", item["original_object"])

    comparison = item["comparison"]

    print("Match status:", comparison["match_status"])
    print("Best match:", comparison["best_match"])
    print("Top matches:", comparison["top_matches"])
