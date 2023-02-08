import sys
import time
import tqdm
import torch
import numpy as np
from torch import nn
from collections import Counter
from torch.utils.data import Dataset, DataLoader


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
    def __init__(self, num_embeddings: int, embedding_dim: int, seq_len: int):
        super().__init__()

        seq_len *= embedding_dim

        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.linear1 = nn.Linear(seq_len, embedding_dim)
        self.linear2 = nn.Linear(embedding_dim, num_embeddings, bias=False)

    def forward(self, x: torch.Tensor):
        x = self.embedding(x)
        x = torch.flatten(x, 1)
        x = self.linear1(x)
        x = torch.tanh(x)
        x = self.linear2(x)
        return x


class BioLMDataset(Dataset):
    def __init__(self, corpus: list, seq_len: int):
        self.seq_len = seq_len

        self.corpus = [np.array(bio, dtype=np.int64) for bio in corpus]
        self.lengths = [len(bio) - seq_len for bio in self.corpus]
        self.total_length = sum(self.lengths)

        for i, bio_len in enumerate(self.lengths):
            assert bio_len > 0

        self.lengths.pop()
        self.lengths = [0] + self.lengths
        self.lengths = np.cumsum(self.lengths)

    def __getitem__(self, item: int):
        idx = np.searchsorted(self.lengths, item, side='right') - 1
        local_idx = item - self.lengths[idx]
        selected_bio = self.corpus[idx]
        x = selected_bio[local_idx:local_idx + self.seq_len]
        y = selected_bio[local_idx + self.seq_len]
        return x, y

    def __len__(self):
        return self.total_length


def encode(fname: str,
           count_threshold: int,
           length_threshold: int,
           vocab: dict = None):
    special_tok = ['<start_bio>', '<end_bio>', '[FAKE]', '[REAL]']
    with open(fname, 'rt') as f:
        tokens = f.read().strip().split()

    if vocab is None:
        counts = Counter(tokens)
        counts = {k: v for k, v in counts.items() if v > count_threshold}
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
            if len(corpus[-1]) <= length_threshold:
                corpus.pop()
            end = True
        elif token in special_tok[-2:]:
            labels.append(token)
        else:
            token = vocab.get(token, None) or vocab['<unk>']
            corpus[-1].append(token)

    return vocab, corpus, labels


def train_categorical(model: nn.Module,
                      optim: torch.optim.Optimizer,
                      train_loader: DataLoader,
                      valid_loader: DataLoader,
                      epochs: int,
                      device: str):
    model = model.to(device)
    criterion = nn.CrossEntropyLoss().to(device)
    print('training started')
    total_step = 0
    for epoch in range(1, epochs + 1):
        total_loss = torch.tensor(0., device=device)
        t = time.time()
        for i, (x, y) in tqdm.tqdm(enumerate(train_loader),
                                   total=len(train_loader)):
            total_step += 1
            x, y = x.to(device), y.to(device)
            optim.zero_grad(set_to_none=True)
            pred = model(x)
            loss = criterion(pred, y)
            loss.backward()
            optim.step()

            # TODO: accuracy calculation

            with torch.inference_mode():
                total_loss += loss.detach()
        with torch.inference_mode():
            total_loss = float(total_loss.cpu())
            total_loss /= len(train_loader)

            # TODO: validation

            print(f'epoch {epoch}:',
                  f'loss {total_loss},',
                  f'time {time.time() - t}')


def main():
    # TODO: argparse
    seq_len = 5
    embedding_dim = 32
    batch_size = 32
    epochs = 50
    model_type = 'FFNN'

    # device detection
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('using device', device)
    if device != 'cuda':
        print('WARNING: cuda not detected', file=sys.stderr)

    # read tokens, init sequences, dataset, and loader
    vocab, train_corpus, train_labels = encode('./mix.train.txt', 3, seq_len)
    _, valid_corpus, valid_labels = encode('./mix.valid.txt', -1, seq_len, vocab=vocab)
    # _, test_corpus, test_labels = encode('./mix.test.txt', -1, seq_len, vocab=vocab)

    train_dataset = BioLMDataset(train_corpus, seq_len)
    train_loader = DataLoader(train_dataset,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=4,
                              pin_memory=True)
    print('training dataset loaded with length', len(train_dataset))

    valid_dataset = BioLMDataset(valid_corpus, seq_len)
    valid_loader = DataLoader(valid_dataset,
                              batch_size=batch_size,
                              shuffle=False,
                              num_workers=2,
                              pin_memory=True)
    print('validation dataset loaded with length', len(valid_dataset))

    # instantiate model and optimizer
    if model_type == 'FFNN':
        model = FFNN(num_embeddings=len(vocab),
                     embedding_dim=embedding_dim,
                     seq_len=seq_len)
    else:
        raise NotImplementedError
    optim = torch.optim.Adam(model.parameters(), lr=1e-3)

    # start training for categorical prediction
    train_categorical(model, optim, train_loader, valid_loader, epochs, device)


if __name__ == '__main__':
    main()
