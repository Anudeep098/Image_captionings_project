import json
import nltk
from collections import Counter
import argparse
import os
nltk.download('punkt')

class Vocabulary:
    def __init__(self):
        self.word2idx = {}
        self.idx2word = {}
        self.idx = 0

    def add_word(self, word):
        if word not in self.word2idx:
            self.word2idx[word] = self.idx
            self.idx2word[self.idx] = word
            self.idx += 1

    def __call__(self, word):
        return self.word2idx.get(word, self.word2idx["<unk>"])

    def __len__(self):
        return len(self.word2idx)
    
    def save_to_json(self, vocab_path):
        with open(vocab_path, 'w') as f:
            json.dump(self.word2idx, f)


def build_vocab(caption_file, threshold):
    with open(caption_file, 'r') as f:
        data = json.load(f)

    counter = Counter()
    for annot in data['annotations']:
        caption = annot['caption']
        tokens = nltk.tokenize.word_tokenize(caption.lower())
        counter.update(tokens)

    words = [word for word, cnt in counter.items() if cnt >= threshold]

    vocab = Vocabulary()
    vocab.add_word("<pad>")
    vocab.add_word("<start>")
    vocab.add_word("<end>")
    vocab.add_word("<unk>")

    for word in words:
        vocab.add_word(word)

    return vocab

def main(args):
    vocab = build_vocab(args.caption_path, args.threshold)
    print(f"Total vocabulary size: {len(vocab)}")
    os.makedirs(os.path.dirname(args.vocab_path), exist_ok=True)
    vocab.save_to_json(args.vocab_path)
    print(f"Vocabulary saved to {args.vocab_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--caption_path', type=str, required=True, help='Path to cleaned caption JSON')
    parser.add_argument('--vocab_path', type=str, required=True, help='Path to save vocabulary JSON')
    parser.add_argument('--threshold', type=int, default=1, help='Minimum word frequency')
    args = parser.parse_args()
    main(args)