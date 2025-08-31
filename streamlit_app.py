import streamlit as st
import torch
from torchvision import transforms
from PIL import Image
import nltk
import os

from caption_model import EncoderCNN, DecoderRNN
from utils.vocab import Vocabulary

nltk.download('punkt')

# --- Load model and vocabulary ---
@st.cache_resource
def load_models(model_path, vocab_path, device):
    # Load vocabulary
    vocab = Vocabulary()
    vocab = vocab.load_from_json(vocab_path)
    vocab_size = len(vocab)

    # Initialize models
    embed_size = 256
    hidden_size = 512
    encoder = EncoderCNN(embed_size).to(device)
    decoder = DecoderRNN(embed_size, hidden_size, vocab_size).to(device)

    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    encoder.load_state_dict(checkpoint['encoder_state_dict'])
    decoder.load_state_dict(checkpoint['decoder_state_dict'])

    encoder.eval()
    decoder.eval()

    return encoder, decoder, vocab

# --- Preprocess image ---
def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406),
                             (0.229, 0.224, 0.225))
    ])
    return transform(image).unsqueeze(0)  # Add batch dimension

# --- Generate caption ---
def generate_caption(image_tensor, encoder, decoder, vocab, device):
    with torch.no_grad():
        image_tensor = image_tensor.to(device)
        features = encoder(image_tensor)
        sampled_ids = decoder.sample(features)

        # DEBUG: show sampled token IDs
        print("Sampled token IDs:", sampled_ids)

        if isinstance(sampled_ids, torch.Tensor):
            sampled_ids = sampled_ids[0].cpu().numpy()

        words = []
        for word_id in sampled_ids:
            word = vocab.idx2word.get(int(word_id), '<unk>')
            if word == '<end>':
                break
            if word != '<start>':
                words.append(word)

        sentence = ' '.join(words)
        if not words:
            sentence = "⚠️ Model did not generate a valid caption."
        return sentence

# --- Streamlit UI ---
def main():
    st.set_page_config(page_title="🖼️ Image Captioning", layout="centered")
    st.title("🖼️ Image Captioning App")
    st.markdown("Upload an image and generate a caption using your trained model.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model and vocabulary
    model_path = "caption_model_epoch_5.pth"
    vocab_path = "data/vocab.json"
    encoder, decoder, vocab = load_models(model_path, vocab_path, device)

    # Upload image
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", use_container_width=True)

        image_tensor = preprocess_image(image)

        if st.button("Generate Caption"):
            caption = generate_caption(image_tensor, encoder, decoder, vocab, device)
            st.success(f"**Caption:** {caption}")

if __name__ == "__main__":
    main()