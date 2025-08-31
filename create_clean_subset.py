import json
import os
import random
from collections import defaultdict

# Load original captions file
INPUT_JSON = 'data/annotations/captions_train2017.json'
OUTPUT_JSON = 'data/annotations/captions_subset_cleaned.json'

# How many images to keep (adjust as needed)
NUM_IMAGES = 1000
CAPTIONS_PER_IMAGE = 5

# Load full dataset
with open(INPUT_JSON, 'r') as f:
    data = json.load(f)

annotations = data['annotations']
images = data['images']

# Group captions by image_id
caption_dict = defaultdict(list)
for ann in annotations:
    caption = ann['caption'].strip()
    # Basic cleaning: remove blank or meaningless captions
    if len(caption) < 5 or caption.lower() in ['.', 'a.', 'a man.', 'a woman.']:
        continue
    caption_dict[ann['image_id']].append(caption)

# Filter image_ids that have enough clean captions
valid_image_ids = [img_id for img_id, caps in caption_dict.items() if len(caps) >= CAPTIONS_PER_IMAGE]

# Take a random sample of image_ids
sampled_ids = random.sample(valid_image_ids, min(NUM_IMAGES, len(valid_image_ids)))

# Create subset annotations
subset_annotations = []
subset_images = []

for img in images:
    if img['id'] in sampled_ids:
        subset_images.append(img)
        captions = caption_dict[img['id']][:CAPTIONS_PER_IMAGE]
        for i, caption in enumerate(captions):
            subset_annotations.append({
                "image_id": img['id'],
                "id": len(subset_annotations),
                "caption": caption
            })

# Create cleaned subset JSON
subset_data = {
    "info": data.get("info", {}),
    "licenses": data.get("licenses", []),
    "images": subset_images,
    "annotations": subset_annotations
}

# Save to new file
os.makedirs(os.path.dirname(OUTPUT_JSON), exist_ok=True)
with open(OUTPUT_JSON, 'w') as f:
    json.dump(subset_data, f)

print(f"✅ Cleaned subset created at {OUTPUT_JSON} with {len(subset_images)} images and {len(subset_annotations)} captions.")
