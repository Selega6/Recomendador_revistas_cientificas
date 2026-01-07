from collections import Counter

import numpy as np
import torch
from torch.utils.data import Dataset


class JournalDataset(Dataset):
    def __init__(self, texts, labels, vocab, max_len=200):
        self.texts = texts.reset_index(drop=True)
        self.labels = labels.reset_index(drop=True)
        self.vocab = vocab
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx]).split()
        ids = [self.vocab.get(word, 1) for word in text][:self.max_len]
        ids += [0] * (self.max_len - len(ids))
        
        return {
            'ids': torch.tensor(ids, dtype=torch.long),
            'label': torch.tensor(self.labels[idx], dtype=torch.long)
        }

def build_vocab(texts, max_features=10000):
    all_words = " ".join(texts).split()
    word_counts = Counter(all_words)
    
    vocab = {word: i+2 for i, (word, _) in enumerate(word_counts.most_common(max_features))}
    vocab['<PAD>'] = 0
    vocab['<UNK>'] = 1
    return vocab