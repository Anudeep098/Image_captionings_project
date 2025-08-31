import os
import urllib.request
import zipfile

def download_and_extract(url, dest_folder):
    os.makedirs(dest_folder, exist_ok=True)
    zip_path = os.path.join(dest_folder, "temp.zip")
    print(f"Downloading from {url}...")
    urllib.request.urlretrieve(url, zip_path)
    print("Extracting...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(dest_folder)
    os.remove(zip_path)
    print(f"✅ Done: {dest_folder}")

def main():
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data'))
    download_and_extract('http://images.cocodataset.org/annotations/annotations_trainval2017.zip',
                         os.path.join(base_dir, 'annotations'))
    download_and_extract('http://images.cocodataset.org/zips/train2017.zip',
                         os.path.join(base_dir, 'images'))
    download_and_extract('http://images.cocodataset.org/zips/val2017.zip',
                         os.path.join(base_dir, 'images'))

if __name__ == '__main__':
    main()
