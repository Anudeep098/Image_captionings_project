import torch
import torch.nn as nn
import torchvision.models as models

class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad = False  # Freeze all ResNet layers

        self.resnet = nn.Sequential(*list(resnet.children())[:-1])  # remove last FC layer
        self.linear = nn.Linear(resnet.fc.in_features, embed_size)
        self.bn = nn.BatchNorm1d(embed_size, momentum=0.01)

    def forward(self, images):
        features = self.resnet(images)         # shape: (batch, 2048, 1, 1)
        features = features.view(features.size(0), -1)  # shape: (batch, 2048)
        features = self.linear(features)       # shape: (batch, embed_size)
        features = self.bn(features)           # shape: (batch, embed_size)
        return features

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        super(DecoderRNN, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, vocab_size)

    def forward(self, features, captions, lengths):  # ✅ Accept all 3 args
        embeddings = self.embed(captions[:, :-1])  # remove <end> token
        embeddings = torch.cat((features.unsqueeze(1), embeddings), 1)
        packed = nn.utils.rnn.pack_padded_sequence(embeddings, lengths, batch_first=True, enforce_sorted=False)
        hiddens, _ = self.lstm(packed)
        outputs = self.linear(hiddens[0])
        return outputs

    def sample(self, features, max_len=20):
        sampled_ids = []
        inputs = features.unsqueeze(1)
        states = None

        for _ in range(max_len):
            hiddens, states = self.lstm(inputs, states)           # (batch, 1, hidden_size)
            outputs = self.linear(hiddens.squeeze(1))             # (batch, vocab_size)
            predicted = outputs.argmax(1)                         # (batch)
            sampled_ids.append(predicted.item())
            inputs = self.embed(predicted).unsqueeze(1)           # (batch, 1, embed)

        return sampled_ids