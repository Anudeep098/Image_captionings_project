import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
import nltk
from utils.dataset import get_loader
from utils.vocab import Vocabulary
from caption_model import EncoderCNN, DecoderRNN
nltk.download('punkt')

# Configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
embed_size = 256
hidden_size = 512
num_layers = 1
num_epochs = 10
batch_size = 32
learning_rate = 1e-3
log_step = 10

def main():
# Load vocabulary
    vocab_path = './data/vocab.json'
    vocab = Vocabulary()
    vocab = vocab.load_from_json(vocab_path)  # <-- this line is critical
    print(f"✅ Vocabulary loaded with {len(vocab)} tokens.")

    # Image transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406),
                             (0.229, 0.224, 0.225))
    ])

    # Data loader
    data_loader = get_loader(
        root='./data/train2017',
        caption_path='./data/annotations/captions_subset_cleaned.json',
        vocab=vocab,
        transform=transform,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0
    )

    # Models
    encoder = EncoderCNN(embed_size).to(device)
    decoder = DecoderRNN(embed_size, hidden_size, len(vocab), num_layers).to(device)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    params = list(decoder.parameters()) + list(encoder.linear.parameters()) + list(encoder.bn.parameters())
    optimizer = optim.Adam(params, lr=learning_rate)

    total_step = len(data_loader)
    for epoch in range(num_epochs):
        for i, (images, captions, lengths) in enumerate(data_loader):
            images = images.to(device)
            captions = captions.to(device)

            targets = nn.utils.rnn.pack_padded_sequence(captions, lengths, batch_first=True, enforce_sorted=False)[0]
            features = encoder(images)
            outputs = decoder(features, captions, lengths)
            loss = criterion(outputs, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if i % log_step == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}], Step [{i}/{total_step}], Loss: {loss.item():.4f}")

        # Save checkpoint
        torch.save({
            'encoder_state_dict': encoder.state_dict(),
            'decoder_state_dict': decoder.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()
        }, f'caption_model_epoch_{epoch+1}.pth')
        print(f"✅ Saved model: caption_model_epoch_{epoch+1}.pth")

if __name__ == '__main__':
    main()