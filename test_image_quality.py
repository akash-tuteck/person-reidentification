from app.services.image_quality_service import ImageQualityService

IMAGE_PATH = (
    "app/uploads/persons/"
    "e75df36b-4b84-416f-97a4-b96c219fe9cc/"
    "person_1.jpg"
)

result = ImageQualityService.validate_crop(IMAGE_PATH)

print("\nIMAGE QUALITY RESULT")
print("--------------------")
print(result)