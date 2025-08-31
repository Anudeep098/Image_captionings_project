import torch

checkpoint_path = 'caption_model_epoch_5.pth'  # Change this to test others
checkpoint = torch.load(checkpoint_path, map_location='cpu')
decoder_state_dict = checkpoint['decoder_state_dict']
embed_weights_shape = decoder_state_dict['embed.weight'].shape
linear_weights_shape = decoder_state_dict['linear.weight'].shape

print(f"📦 Checking {checkpoint_path}")
print(f"embed.weight shape: {embed_weights_shape}")
print(f"linear.weight shape: {linear_weights_shape}")
