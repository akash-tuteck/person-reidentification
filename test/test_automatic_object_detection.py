import json

from app.services.grounding_dino_service import GroundingDINOService


IMAGE_ID = "d0e0c0f5-4c0c-4a66-9627-afbf15461169"


def main():
    print("Testing Gemma + Grounding DINO automatic detection...")
    result = GroundingDINOService.detect_objects_automatically(
        image_id=IMAGE_ID
    )
    print(json.dumps(result, indent=4, ensure_ascii=False))

    discovery = result.get("discovery", {})
    print("\nDISCOVERED CATEGORIES:")
    for item in discovery.get("objects", []):
        print(f"- {item.get('category')}: {item.get('description', '')}")

    detection = result.get("detection")
    if detection is None:
        print("\nNo qualifying objects were discovered.")
        return

    print("\nDETECTION SUMMARY")
    print("Total objects:", detection.get("total_objects", 0))
    print("Category counts:", detection.get("category_counts", {}))
    print("Annotated image:", detection.get("annotated_image_path"))
    print("Metadata:", detection.get("metadata_path"))


if __name__ == "__main__":
    main()
