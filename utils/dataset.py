import os
import json
import torch
import torch.utils.data as data
from PIL import Image
import torchvision.transforms as transforms
import nltk

class CocoDataset(data.Dataset):
    def __init__(self, root, caption_path, vocab, transform=None):
        """
        Args:
            root: Directory with all the images.
            caption_path: Path to the json file with captions.
            vocab: Vocabulary object.
            transform: Optional transform to be applied on an image.
        """
        self.root = root
        self.vocab = vocab
        self.transform = transform

        with open(caption_path, 'r') as f:
            captions = json.load(f)

        self.captions = captions['annotations']
        self.image_id_to_filename = {
            img['id']: img['file_name'] for img in captions['images']
        }

    def __getitem__(self, index):
        caption_data = self.captions[index]
        image_id = caption_data['image_id']
        caption = caption_data['caption']
        image_filename = self.image_id_to_filename[image_id]
        image_path = os.path.join(self.root, image_filename)

        image = Image.open(image_path).convert('RGB')
        if self.transform:
            image = self.transform(image)

        tokens = nltk.tokenize.word_tokenize(caption.lower())
        caption_ids = []
        caption_ids.append(self.vocab('<start>'))
        caption_ids.extend([self.vocab(token) for token in tokens])
        caption_ids.append(self.vocab('<end>'))

        caption_tensor = torch.Tensor(caption_ids).long()

        return image, caption_tensor

    def __len__(self):
        return len(self.captions)


def get_loader(root, caption_path, vocab, transform, batch_size, shuffle, num_workers):
    """Returns torch.utils.data.DataLoader for custom Coco dataset."""
    dataset = CocoDataset(root=root, caption_path=caption_path, vocab=vocab, transform=transform)

    def collate_fn(data):
        data.sort(key=lambda x: len(x[1]), reverse=True)
        images, captions = zip(*data)

        images = torch.stack(images, 0)
        lengths = [len(cap) for cap in captions]
        targets = torch.zeros(len(captions), max(lengths)).long()

        for i, cap in enumerate(captions):
            end = lengths[i]
            targets[i, :end] = cap[:end]

        return images, targets, lengths

    data_loader = torch.utils.data.DataLoader(dataset=dataset,
                                              batch_size=batch_size,
                                              shuffle=shuffle,
                                              num_workers=num_workers,
                                              collate_fn=collate_fn)

    return data_loader
