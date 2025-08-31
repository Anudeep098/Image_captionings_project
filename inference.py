import torch
import torchvision.transforms as transforms
from PIL import Image
import argparse
import nltk
nltk.download('punkt')

from utils.vocab import Vocabulary
from caption_model import EncoderCNN, DecoderRNN


def load_image(image_path, transform=None):
    image = Image.open(image_path).convert('RGB')
    if transform:
        image = transform(image).unsqueeze(0)
    return image


def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load vocabulary
    vocab = Vocabulary()
    vocab = vocab.load_from_json(args.vocab_path)
    print(f"✅ Vocabulary loaded with {len(vocab)} tokens.")
    if len(vocab) <= 4:
        print("❌ ERROR: Vocabulary is too small.")
        return
    print(f"🔎 Sample tokens from vocab: {list(vocab.word2idx.keys())[:10]}")

    # Image preprocessing
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406),
                             (0.229, 0.224, 0.225))
    ])
    image = load_image(args.image, transform).to(device)

    # Model parameters
    embed_size = 256
    hidden_size = 512

    # Initialize models
    encoder = EncoderCNN(embed_size).to(device)
    decoder = DecoderRNN(embed_size, hidden_size, len(vocab)).to(device)

    # Load weights
    checkpoint = torch.load(args.model_path, map_location=device)
    if 'encoder_state_dict' in checkpoint and 'decoder_state_dict' in checkpoint:
        encoder.load_state_dict(checkpoint['encoder_state_dict'])
        decoder.load_state_dict(checkpoint['decoder_state_dict'])
    elif 'encoder' in checkpoint and 'decoder' in checkpoint:
        encoder.load_state_dict(checkpoint['encoder'])
        decoder.load_state_dict(checkpoint['decoder'])
    else:
        print("❌ ERROR: Unknown checkpoint format.")
        return

    encoder.eval()
    decoder.eval()

    # Generate caption
    with torch.no_grad():
        features = encoder(image)
        sampled_ids = decoder.sample(features)

        if isinstance(sampled_ids, torch.Tensor):
            sampled_ids = sampled_ids[0].cpu().numpy()

        print(f"🧪 Sampled token IDs: {sampled_ids}")

        words = []
        for word_id in sampled_ids:
            word = vocab.idx2word.get(int(word_id), '<unk>')
            if word == '<end>':
                break
            if word != '<start>':
                words.append(word)

        sentence = ' '.join(words)
        unk_count = sum(1 for w in words if w == '<unk>')

        print(f"\n🖼️ Predicted Caption:\n{sentence}")
        print(f"🔍 Unknown token count: {unk_count}/{len(words)}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', type=str, required=True)
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--vocab_path', type=str, required=True)
    args = parser.parse_args()
    main(args)