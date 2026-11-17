
from app.services.object_feature_service import ObjectFeatureService

image_id = "95ce8ac8-d586-495c-bd86-48a9c9389f33"

result = ObjectFeatureService.extract_features(image_id)

print("\nOBJECT FEATURE EXTRACTION RESULT")
print("--------------------------------")

print("Image ID:", result["image_id"])
print("Total processed:", result["total_objects_processed"])
print("Category counts:", result["category_counts"])
print("Embedding dimension:", result["embedding_dimension"])
print("Model:", result["model"])
print("Status:", result["status"])

for obj in result["processed_objects"]:
    print(obj)
