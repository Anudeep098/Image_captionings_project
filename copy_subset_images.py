import os
import shutil
import argparse
import json

def copy_subset_images(json_path, src_folder, dst_folder):
    # Load annotations
    with open(json_path, 'r') as f:
        data = json.load(f)

    # Make sure destination exists
    os.makedirs(dst_folder, exist_ok=True)

    # Collect image file names
    image_ids = set()
    for item in data['annotations']:
        image_ids.add(item['image_id'])

    print(f"Total unique images in subset: {len(image_ids)}")

    for image in data['images']:
        if image['id'] in image_ids:
            src_path = os.path.join(src_folder, image['file_name'])
            dst_path = os.path.join(dst_folder, image['file_name'])

            if os.path.exists(src_path):
                shutil.copy2(src_path, dst_path)
            else:
                print(f"Warning: {image['file_name']} not found in source folder.")

    print("✅ Subset image copy completed.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--json_path', required=True, help='Path to the cleaned JSON annotation file')
    parser.add_argument('--src', required=True, help='Source image folder (e.g., data/train2017)')
    parser.add_argument('--dst', required=True, help='Destination folder to copy subset (e.g., data/train2017_subset)')
    args = parser.parse_args()

    copy_subset_images(args.json_path, args.src, args.dst)