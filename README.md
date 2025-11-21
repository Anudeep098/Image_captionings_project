# Image Captioning Project

This project generates captions for images using a deep learning model.

 Project Structure:
- train.py — training script
- inference.py — run the model for predictions
- utils/ — helper functions and datasets
- sample.jpg — sample image
- vocab.pkl — vocabulary file
- build_vocab.py` — Builds vocabulary from captions dataset  
- caption_model.py — Defines the image captioning model  
- check_model.py — Checks if the model loads correctly  
- check_vocab_size.py — Verifies vocabulary size  
- copy_subset_images.py — Copies a subset of images  
- create_clean_subset.py — Cleans and prepares dataset  
- create_subset.py — Creates a smaller dataset subset  
- inference.py — Generates captions for images  
- model.py — Model helper functions  
- streamlit_app.py — Web app for generating captions  
- train.py — Trains the image captioning model

## How to Run
1. Install dependencies: pip install -r requirements.txt
2. Run Train: python train.py
3. Run inference: python inference.py
4. For Output: streamlit run streamlit_app.py

🛠️ Technologies Used

Here are the main technologies and tools I used to build this Image Captioning project:

🐍 Python – Core programming language

🔥 PyTorch – For building and training the deep learning model

🧠 CNN (ResNet / VGG) – Used for extracting image features

✍️ RNN / LSTM – Used for generating captions word by word

🏋️ TorchVision – For loading pretrained image models

🔡 NLTK / Text Processing – Used for cleaning text and creating vocabulary

📦 Pickle (pkl) – For saving the vocabulary

🖼️ PIL / OpenCV – For handling and preprocessing images

📊 Matplotlib – For visualizing samples (optional)

🌐 Streamlit – For creating a simple web app to test the model

🗂️ COCO / Flickr8k Dataset – Dataset used for training captions

🧰 NumPy & Pandas – For data handling and processing

