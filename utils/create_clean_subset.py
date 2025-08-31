import json
import os

subset_image_dir = "data/train_subset"
original_anno_path = "data/annotations/captions_train2017.json"
cleaned_anno_path = "data/annotations/captions_subset_cleaned.json"

# Load original captions
with open(original_anno_path, "r") as f:
    annotations = json.load(f)

# Get list of image filenames in subset
subset_filenames = set(os.listdir(subset_image_dir))
subset_image_ids = {
    int(fname.split(".")[0]) for fname in subset_filenames if fname.endswith(".jpg")
}

# Filter images and annotations
filtered_images = [img for img in annotations["images"] if img["id"] in subset_image_ids]
filtered_annotations = [ann for ann in annotations["annotations"] if ann["image_id"] in subset_image_ids]

# Create new JSON structure
cleaned_data = {
    "images": filtered_images,
    "annotations": filtered_annotations,
    "type": "captions",
    "licenses": annotations.get("licenses", []),
    "info": annotations.get("info", {})
}

# Save cleaned annotations
os.makedirs(os.path.dirname(cleaned_anno_path), exist_ok=True)
with open(cleaned_anno_path, "w") as f:
    json.dump(cleaned_data, f)

print(f"Saved cleaned annotations to {cleaned_anno_path}")
