import torch
import numpy as np
from torch import nn
from collections import Counter
from torch.nn import functional as F


class Token:
    __slots__ = ['tok', 'idx', 'count']

    def __init__(self, tok, idx, count=0):
        self.tok = tok
        self.idx = idx
        self.count = count

    def __int__(self):
        return self.idx

    def __str__(self):
        return self.tok

    def __repr__(self):
        return f"Token('{self.tok}', {self.idx}, {self.count})"


class FFNN(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, seq_len, device):
        super().__init__()
        self.device = device

        seq_len *= embedding_dim

        self.in_embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.linear1 = nn.Linear(seq_len, num_embeddings)
        self.linear2 = nn.Linear(embedding_dim, num_embeddings, bias=False)

    def forward(self, x):
        x = self.as_vec(x)
        x = torch.flatten(x, 1)
        x = self.linear1(x)
        x = F.tanh(x)
        x = self.linear2(x)
        return x

    def as_vec(self, x):
        x = np.array(x, dtype=np.int64)
        x = self.embedding(x)
        return x


def encode(fname, threshold, vocab=None):
    special_tok = ['<start_bio>', '<end_bio>', '[FAKE]', '[REAL]']
    with open(fname, 'rt') as f:
        tokens = f.read().strip().split()

    if vocab is None:
        counts = Counter(tokens)
        counts = {k: v for k, v in counts.items() if v >= threshold}
        for tok in special_tok:
            counts.pop(tok, None)
        vocab = {k: Token(k, i, v)
                 for i, (k, v) in enumerate(counts.items())}

        unk_tok = Token('<unk>', len(vocab), 100)
        vocab[unk_tok.tok] = unk_tok

    corpus = []
    labels = []
    end = True

    for token in tokens:
        if special_tok[0] in token:
            if end:
                corpus.append([])
            else:
                corpus[-1] = []
            end = False
        elif special_tok[1] in token:
            end = True
        elif token in special_tok[-2:]:
            labels.append(token)
        else:
            token = vocab.get(token, None) or vocab['<unk>']
            corpus[-1].append(token)

    return vocab, corpus, labels


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    vocab, train_corpus, train_labels = encode('./mix.train.txt', 3)
    # _, _, test_corpus, test_labels = encode('./mix.test.txt', -1, vocab)
    # print(vocab)
    # print(len(vocab))
    # print(corpus)
    print(len(train_corpus))
    print(len(train_labels))


if __name__ == '__main__':
    main()
