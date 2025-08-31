from torch.utils.data import Dataset
from PIL import Image
import os
import json

class CocoDataset(Dataset):
    def __init__(self, root_folder, annotation_file, transform=None, vocab=None):
        self.root_folder = root_folder
        self.transform = transform
        self.vocab = vocab

        with open(annotation_file, 'r') as f:
            data = json.load(f)

        self.annotations = data['annotations']
        self.image_id_to_filename = {
            img['id']: img['file_name'] for img in data['images']
        }

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        annotation = self.annotations[index]
        img_id = annotation['image_id']
        caption = annotation['caption']
        image_path = os.path.join(self.root_folder, self.image_id_to_filename[img_id])
        
        image = Image.open(image_path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)

        tokens = self.vocab.tokenize(caption)
        caption_ids = [self.vocab('<start>')] + [self.vocab(token) for token in tokens] + [self.vocab('<end>')]
        caption_tensor = torch.tensor(caption_ids)

        return image, caption_tensor
