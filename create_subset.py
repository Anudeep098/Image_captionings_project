import json
import random

FULL_ANNOTATION_PATH = './data/annotations/captions_train2017.json'
SUBSET_ANNOTATION_PATH = './data/annotations/captions_subset.json'
NUM_IMAGES = 1000  # You can change this

with open(FULL_ANNOTATION_PATH, 'r') as f:
    coco_data = json.load(f)

image_ids = list({ann["image_id"] for ann in coco_data["annotations"]})
subset_image_ids = set(random.sample(image_ids, NUM_IMAGES))

# Subset images and annotations
subset_images = [img for img in coco_data["images"] if img["id"] in subset_image_ids]
subset_annotations = [ann for ann in coco_data["annotations"] if ann["image_id"] in subset_image_ids]

subset_data = {
    "info": coco_data.get("info", {}),
    "licenses": coco_data.get("licenses", []),
    "images": subset_images,
    "annotations": subset_annotations
}

with open(SUBSET_ANNOTATION_PATH, 'w') as f:
    json.dump(subset_data, f)

print(f"✅ Subset saved to {SUBSET_ANNOTATION_PATH} with {len(subset_images)} images.")
