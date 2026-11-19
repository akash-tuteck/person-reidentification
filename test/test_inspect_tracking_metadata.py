
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

VIDEO_ID = "b4b6769d-120d-4f32-bfd6-9e64da402293"

BASE_DIR = PROJECT_ROOT / "app" / "uploads"

FILES = {
    "PERSON TRACKING": (
        BASE_DIR
        / "video_tracking"
        / VIDEO_ID
        / "tracking_metadata.json"
    ),
    "OBJECT TRACKING": (
        BASE_DIR
        / "video_object_tracking"
        / VIDEO_ID
        / "object_tracking_metadata.json"
    )
}


def inspect_file(name, path):
    print("\n" + "=" * 60)
    print(name)
    print("=" * 60)

    if not path.exists():
        print("FILE NOT FOUND:", path)
        return

    with open(path, "r", encoding="utf-8") as file:
        data = json.load(file)

    print("\nTOP-LEVEL KEYS:")
    print(list(data.keys()))

    print("\nSUMMARY:")
    for key, value in data.items():
        if isinstance(value, list):
            print(f"{key}: list with {len(value)} items")
        elif isinstance(value, dict):
            print(f"{key}: dictionary with {len(value)} keys")
        else:
            print(f"{key}: {value}")

    for key, value in data.items():
        if isinstance(value, list) and value:
            print(f"\nFIRST ITEM FROM '{key}':")
            print(json.dumps(value[0], indent=4)[:5000])

        elif isinstance(value, dict) and value:
            first_key = next(iter(value))
            print(f"\nFIRST ITEM FROM '{key}' ({first_key}):")
            print(
                json.dumps(value[first_key], indent=4)[:5000]
            )


def main():
    for name, path in FILES.items():
        inspect_file(name, path)


if __name__ == "__main__":
    main()
