
# app/services/object_categories.py

"""
Shared COCO object category configuration.

COCO contains 80 classes:
- 1 human category: person (class ID 0)
- 79 non-human object categories

This configuration is shared by image and video object detection.
"""

COCO_CLASS_NAMES = [
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "backpack",
    "umbrella",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "couch",
    "potted plant",
    "bed",
    "dining table",
    "toilet",
    "tv",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
]

PERSON_CLASS_ID = 0

# Category name -> YOLO COCO class ID
NON_PERSON_COCO_CLASS_IDS = {
    category: class_id
    for class_id, category in enumerate(COCO_CLASS_NAMES)
    if class_id != PERSON_CLASS_ID
}

NON_PERSON_CATEGORIES = list(NON_PERSON_COCO_CLASS_IDS.keys())

NON_PERSON_CLASS_IDS = list(NON_PERSON_COCO_CLASS_IDS.values())

# Shared settings for image and video object detection
OBJECT_DETECTION_CONFIDENCE = 0.25
OBJECT_DETECTION_IMAGE_SIZE = 640


def validate_object_categories():
    """Validate the shared COCO category configuration."""

    assert len(COCO_CLASS_NAMES) == 80
    assert len(NON_PERSON_COCO_CLASS_IDS) == 79
    assert "person" not in NON_PERSON_COCO_CLASS_IDS
    assert PERSON_CLASS_ID == 0

    return True
