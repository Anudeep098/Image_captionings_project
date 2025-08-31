import json
import nltk
from collections import Counter
from nltk.tokenize import word_tokenize


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
        return self.word2idx.get(word, self.word2idx['<unk>'])

    def __len__(self):
        return len(self.word2idx)

    def save_to_json(self, vocab_path):
        with open(vocab_path, 'w', encoding='utf-8') as f:
            json.dump({
                'word2idx': self.word2idx,
                'idx2word': self.idx2word,
                'idx': self.idx
            }, f, ensure_ascii=False, indent=4)

    @classmethod
    def load_from_json(cls, filepath):
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)

        if isinstance(data, dict) and 'word2idx' in data:
            vocab = cls()
            vocab.word2idx = data['word2idx']
            vocab.idx2word = {int(k): v for k, v in data['idx2word'].items()}
            vocab.idx = data['idx']
            print("✅ Loaded full vocabulary with", len(vocab.word2idx), "tokens.")
            print("🔎 Sample idx2word:", list(vocab.idx2word.items())[:10])
            return vocab

        elif isinstance(data, dict):
            vocab = cls()
            vocab.word2idx = data
            vocab.idx2word = {idx: word for word, idx in data.items()}
            vocab.idx = len(data)
            print("✅ Loaded simple word2idx-only vocab with", len(vocab.word2idx), "tokens.")
            print("🔎 Sample idx2word:", list(vocab.idx2word.items())[:10])
            return vocab

        else:
            raise ValueError("❌ vocab.json format is invalid. Expected {word2idx, idx2word, idx} keys or {word: index} mapping.")


def build_vocab(captions, threshold):
    counter = Counter()
    for caption in captions:
        tokens = word_tokenize(caption.lower())
        counter.update(tokens)

    words = [word for word, cnt in counter.items() if cnt >= threshold]

    vocab = Vocabulary()
    # Add special tokens
    vocab.add_word('<pad>')
    vocab.add_word('<start>')
    vocab.add_word('<end>')
    vocab.add_word('<unk>')

    for word in words:
        vocab.add_word(word)

    return vocab
