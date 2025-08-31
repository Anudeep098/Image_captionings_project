import os
import torch

for file in os.listdir():
    if file.endswith(".pth"):
        checkpoint = torch.load(file, map_location='cpu')
        try:
            decoder_state_dict = checkpoint['decoder_state_dict']
            vocab_size = decoder_state_dict['embed.weight'].shape[0]
            print(f"{file}: vocab size = {vocab_size}")
        except:
            print(f"{file}: ❌ Invalid format")
