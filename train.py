import sys
import time
import tqdm
import torch
import argparse
import numpy as np
from torch import nn
from copy import deepcopy
from typing import Callable
from collections import Counter
from torch.utils.data import Dataset, DataLoader, default_collate
import matplotlib.pyplot as plt


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


class SequenceModel(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, pad: bool = False):
        super().__init__()
        self.in_embedding = nn.Embedding(num_embeddings + int(pad),
                                         embedding_dim,
                                         padding_idx=num_embeddings if pad else None)
        self.out_embedding = nn.Linear(embedding_dim, num_embeddings, bias=False)

    def init_params(self):
        def initialize(m: nn.Module):
            if isinstance(m, nn.Embedding):
                nn.init.xavier_normal_(m.weight)
            elif type(m) in [nn.LSTM, nn.LSTMCell]:
                for name, param in m.named_parameters():
                    if 'weight' in name:
                        nn.init.xavier_normal_(param)
                    elif 'bias' in name:
                        nn.init.zeros_(param)

        self.apply(initialize)


class FFNN(SequenceModel):
    def __init__(self, num_embeddings: int, embedding_dim: int, seq_len: int):
        super().__init__(num_embeddings, embedding_dim)

        seq_len *= embedding_dim
        self.linear1 = nn.Linear(seq_len, embedding_dim)

        self.init_params()
        self.out_embedding.weight = self.in_embedding.weight

    def forward(self, x: torch.Tensor):
        x = self.in_embedding(x)
        x = torch.flatten(x, 1)
        x = self.linear1(x)
        x = torch.tanh(x)
        x = self.out_embedding(x)
        return x


class LSTM(SequenceModel):
    def __init__(self, num_embeddings: int, embedding_dim: int, num_layers: int):
        super().__init__(num_embeddings, embedding_dim, pad=True)
        self.lstm = nn.LSTM(embedding_dim, embedding_dim, num_layers, batch_first=True)

        self.init_params()
        assert self.out_embedding.weight.shape == self.in_embedding.weight[:-1].shape
        self.out_embedding.weight = nn.Parameter(self.in_embedding.weight[:-1])
        nn.init.zeros_(self.in_embedding.weight[-1])  # pad index

    def forward(self, x):
        x, lengths = x
        x = self.in_embedding(x)
        x = nn.utils.rnn.pack_padded_sequence(x, lengths, batch_first=True)
        x, _ = self.lstm(x)
        x = self.out_embedding(x.data)
        return x


class BioFixedLenDataset(Dataset):
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

    def collate(self, batch):
        return default_collate(batch)


class BioVariableLenDataset(Dataset):
    def __init__(self, corpus, pad_val: int):
        self.corpus = [np.array(bio, dtype=np.int64) for bio in corpus]
        self.pad_val = pad_val

        for bio in self.corpus:
            assert len(bio) > 1

    def __getitem__(self, item):
        return self.corpus[item]

    def __len__(self):
        return len(self.corpus)

    def collate(self, batch: list):
        batch = sorted(batch, key=len, reverse=True)
        lengths = [len(sample) for sample in batch]
        x = np.full((len(batch), len(batch[0])),
                    fill_value=self.pad_val, dtype=np.int64)
        y = deepcopy(x)
        for i, row in enumerate(batch):
            x[i, :len(row) - 1] = row[:-1]
            y[i, :len(row) - 1] = row[1:]
        return (torch.as_tensor(x), lengths), torch.as_tensor(y)


def encode(fnames: list,
           count_threshold: int,
           length_threshold: int,
           vocab: dict = None):
    special_tok = ['<start_bio>', '<end_bio>', '[FAKE]', '[REAL]']
    tokens = []

    # read all files as tokens
    for fname in fnames:
        with open(fname, 'rt', encoding='utf8') as f:
            content = f.read()
            content = content.replace('< start_bio >', special_tok[0])
            content = content.replace('< end_bio >', special_tok[1])
            if 'fake' in fname:
                content = content.replace(special_tok[1], f' {special_tok[1]} {special_tok[2]} ')
            elif 'real' in fname:
                content = content.replace(special_tok[1], f' {special_tok[1]} {special_tok[3]} ')
            tokens += content.strip().split()

    corpus = []
    end = True

    # split to list of sequences, each sequence is a single bio as tokens
    for token in tokens:
        if special_tok[0] in token:
            if end:
                corpus.append([])
            else:
                corpus[-1] = []
            end = False
            continue
        elif special_tok[1] in token:
            if len(corpus[-1]) > length_threshold:
                end = True
        corpus[-1].append(token)

    # deduplicate
    corpus = list(set(tuple(seq) for seq in corpus))

    # count and create vocab
    if vocab is None:
        flat_corpus = [token for seq in corpus for token in seq]
        counts = Counter(flat_corpus)
        counts = {k: v for k, v in counts.items() if v > count_threshold}
        vocab = {k: Token(k, i, v)
                 for i, (k, v) in enumerate(counts.items())}

        unk_tok = Token('<unk>', len(vocab), 100)
        vocab[unk_tok.tok] = unk_tok

    # convert tokens to Token object
    for i, seq in enumerate(corpus):
        temp = []
        for token in seq:
            token = vocab.get(token, None) or vocab['<unk>']
            temp.append(token)
        corpus[i] = temp

    return vocab, corpus


def save_model(model: nn.Module):
    if type(model) == FFNN:
        print("Model is FFN")
        path = "FNN_model.pth"
    elif type(model) == LSTM:
        print("Model is LSTM")
        path = "LSTM_model.pth"
    torch.save(model, path)


def train_categorical(model: nn.Module,
                      optim: torch.optim.Optimizer,
                      criterion: Callable,
                      train_loader: DataLoader,
                      valid_loader: DataLoader,
                      epochs: int,
                      device: str):
    model = model.to(device)
    print('training started')
    total_step = 0
    train_perplexity_per_epoch = []
    valid_perplexity_per_epoch = []
    for epoch in range(1, epochs + 1):
        train_loss = torch.tensor(0., device=device)
        t = time.time()
        model.train()
        for i, (x, y) in tqdm.tqdm(enumerate(train_loader),
                                   total=len(train_loader)):
            total_step += 1
            x, y = x.to(device), y.to(device)
            optim.zero_grad(set_to_none=True)
            pred = model(x)
            loss = criterion(pred, y)
            loss.backward()
            optim.step()

            with torch.inference_mode():
                train_loss += loss.detach()
        with torch.inference_mode():
            model.eval()

            # calculate perplexity for train and valid set
            train_loss = float(train_loss.cpu())
            train_loss /= len(train_loader)
            train_perplexity = np.exp(train_loss)
            train_perplexity_per_epoch.append(train_perplexity)

            valid_loss = torch.tensor(0., device=device)
            for i, (x, y) in enumerate(valid_loader):
                x, y = x.to(device), y.to(device)
                pred = model(x)
                loss = criterion(pred, y)
                valid_loss += loss
            valid_loss /= len(valid_loader)
            valid_loss = float(valid_loss.cpu())
            valid_perplexity = np.exp(valid_loss)
            valid_perplexity_per_epoch.append(valid_perplexity)

            print(f'epoch {epoch}:',
                  f'loss {train_loss},',
                  f'valid_loss {valid_loss},',
                  f'train_perplexity {train_perplexity},',
                  f'valid_perplexity {valid_perplexity},',
                  f'time {time.time() - t}')
    results = {
        'train_perplexity': train_perplexity_per_epoch,
        'valid_perplexity': valid_perplexity_per_epoch,
    }

    save_model(model.cpu())

    return results


def test_categorical(model, test_corpus, seq_len, vocab, device):
    model = model.to(device)
    model.eval()
    test_data = [np.array(bio, dtype=np.int64) for bio in test_corpus]
    # confusion matrix for binary classification
    # TP: true positive (label REAL is predicted correctly)
    TP, FP, FN, TN = 0, 0, 0, 0
    for data in test_data:
        x = torch.tensor(data[-seq_len - 1:-1], device=device)
        y = torch.tensor(data[-1], device=device)
        logits = model(x.unsqueeze(0))
        logits.squeeze_(0)
        if logits[vocab['[FAKE]'].idx] > logits[vocab['[REAL]'].idx]:
            pred = vocab['[FAKE]'].idx
        else:
            pred = vocab['[REAL]'].idx
        if pred == y:
            if pred == vocab['[REAL]'].idx:
                TP += 1
            else:
                TN += 1
        else:
            if pred == vocab['[REAL]'].idx:
                FP += 1
            else:
                FN += 1
    accuracy = (TP + TN) / (TP + FP + FN + TN)
    results = {
        'accuracy': accuracy,
        'confusion_matrix': [[TP, FP], [FN, TN]],
    }
    return results


def parse_arguments():
    parser = argparse.ArgumentParser()
    # todo: add more arguments
    parser.add_argument('-model', type=str, default='FFNN')
    parser.add_argument('-d_model', type=int, default=100)
    parser.add_argument('-d_hidden', type=int, default=100)
    parser.add_argument('-n_layers', type=int, default=2)
    parser.add_argument('-batch_size', type=int, default=32)
    parser.add_argument('-seq_len', type=int, default=5)
    parser.add_argument('-printevery', type=int, default=5000)
    parser.add_argument('-window', type=int, default=3)
    parser.add_argument('-epochs', type=int, default=50)
    parser.add_argument('-lr', type=float, default=1e-3)
    parser.add_argument('-dropout', type=int, default=0.35)
    parser.add_argument('-clip', type=int, default=2.0)

    return parser.parse_args()

def plot_learning_curve(train_perplexity, valid_perplexity, model_type):
    epochs = range(1, len(train_perplexity)+1)
    plt.plot(epochs,train_perplexity, 'b', label='Train')
    plt.plot(epochs,valid_perplexity, 'r', label='Validation')
    plt.xlabel('Epoch')
    plt.ylabel('Perplexity')
    plt.legend()
    plt.savefig(f'perplexity_plot_{model_type}.png')
    plt.show()

def plot_confusion_matrix(confusion_matrix, model_type):
    categories = ['Real', 'Fake']
    fig, ax = plt.subplots()
    im = ax.imshow(confusion_matrix)
    ax.set_xticks(np.arange(len(categories)))
    ax.set_yticks(np.arange(len(categories)))
    ax.set_xticklabels(categories)
    ax.set_yticklabels(categories)
    plt.setp(ax.get_xticklabels(), ha="right",
             rotation_mode="anchor")
    for i in range(len(categories)):
        for j in range(len(categories)):
            text = ax.text(j, i, confusion_matrix[i, j],
                           ha="center", va="center", color="w",fontsize=15, fontweight='bold')
    ax.set_title("Confusion Matrix")
    ax.set_ylabel("Predicted")
    ax.set_xlabel("Actual")
    # Save the figure
    plt.savefig(f'confusion_matrix_{model_type}.png')


def main():
    params = parse_arguments()

    seq_len = params.seq_len
    batch_size = params.batch_size
    epochs = params.epochs
    model_type = params.model
    lr = params.lr
    embedding_dim = 32

    # device detection
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('using device', device)
    if device != 'cuda':
        print('WARNING: cuda not detected', file=sys.stderr)

    # read tokens, init sequences, dataset, and loader
    vocab, train_corpus = encode(['./mix.train.tok', 'fake.train.tok', 'real.train.tok'],
                                 count_threshold=3,
                                 length_threshold=seq_len)
    _, valid_corpus = encode(['./mix.valid.tok', 'fake.valid.tok', 'real.valid.tok'],
                             count_threshold=-1,
                             length_threshold=seq_len,
                             vocab=vocab)
    _, test_corpus = encode(['./mix.test.tok', 'fake.test.tok', 'real.test.tok'],
                            count_threshold=-1,
                            length_threshold=seq_len,
                            vocab=vocab)

    for i, corpus in enumerate([train_corpus, valid_corpus, test_corpus]):
        for seq in corpus:
            assert seq[-1].tok in ['[REAL]', '[FAKE]'], f'{i}: {seq[-20:]}'
            assert seq[-2].tok == '<end_bio>', f'{i}: {seq[-20:]}'
            assert seq[0].tok != '<start_bio>', f'{i}: {seq[-20:]}'

    # instantiate model and model specific parameters
    if model_type == 'FFNN':
        model = FFNN(num_embeddings=len(vocab),
                     embedding_dim=embedding_dim,
                     seq_len=seq_len)
        criterion = nn.CrossEntropyLoss().to(device)
        train_dataset = BioFixedLenDataset(train_corpus, seq_len)
        valid_dataset = BioFixedLenDataset(valid_corpus, seq_len)
    else:
        raise NotImplementedError

    train_loader = DataLoader(train_dataset,
                              batch_size=batch_size,
                              shuffle=True,
                              collate_fn=train_dataset.collate,
                              num_workers=4,
                              pin_memory=True)
    print('training dataset loaded with length', len(train_dataset))

    valid_loader = DataLoader(valid_dataset,
                              batch_size=batch_size,
                              shuffle=False,
                              collate_fn=valid_dataset.collate,
                              num_workers=2,
                              pin_memory=True)
    print('validation dataset loaded with length', len(valid_dataset))

    optim = torch.optim.Adam(model.parameters(), lr=lr)

    # start training for categorical prediction
    results = train_categorical(model=model,
                                optim=optim,
                                criterion=criterion,
                                train_loader=train_loader,
                                valid_loader=valid_loader,
                                epochs=epochs,
                                device=device)
    plot_learning_curve(results['train_perplexity'],results['valid_perplexity'], model_type)
    test_results = test_categorical(model=model,
                                    test_corpus=test_corpus,
                                    seq_len=seq_len,
                                    vocab=vocab,
                                    device=device)
    plot_confusion_matrix(test_results['confusion_matrix'], model_type)


if __name__ == '__main__':
    main()
