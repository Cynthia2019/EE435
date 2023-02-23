import os
import sys
import time
import tqdm
import torch
import random
import pickle
import argparse
import numpy as np
import matplotlib.pyplot as plt

from torch import nn
from typing import Callable
from collections import Counter
from torch.utils.data import Dataset, DataLoader, default_collate
from sklearn.neighbors import KNeighborsClassifier
from scipy.spatial.distance import jensenshannon


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
    def __init__(self, num_embeddings: int, embedding_dim: int,
                 pad: bool = False, device: str = 'cuda'):
        super().__init__()
        self.in_embedding = nn.Embedding(num_embeddings + int(pad),
                                         embedding_dim,
                                         padding_idx=num_embeddings if pad else None)
        self.out_embedding = nn.Linear(embedding_dim, num_embeddings, bias=False)
        self.device = device

    def init_params(self):
        def initialize(m: nn.Module):
            if isinstance(m, nn.Embedding):
                nn.init.xavier_normal_(m.weight)
            # initiate weights by xavier normal and set all bias to zero
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if getattr(m, 'bias', None) is not None:
                    nn.init.zeros_(m.bias)
            elif type(m) in [nn.LSTM, nn.LSTMCell]:
                for name, param in m.named_parameters():
                    if 'weight' in name:
                        nn.init.xavier_normal_(param)
                    elif 'bias' in name:
                        nn.init.zeros_(param)

        self.apply(initialize)


class FFNN(SequenceModel):
    def __init__(self, num_embeddings: int, embedding_dim: int, window: int,
                 drop_ratio: float, device: str):
        super().__init__(num_embeddings, embedding_dim, device=device)

        window *= embedding_dim
        self.linear = nn.Linear(window, embedding_dim)
        self.dropout = nn.Dropout(drop_ratio) if drop_ratio else None

        self.init_params()
        self.out_embedding.weight = self.in_embedding.weight

    def forward(self, x: torch.Tensor):
        x = x.to(self.device)
        x = self.in_embedding(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x) if self.dropout else x
        x = self.linear(x)
        x = torch.tanh(x)
        x = self.out_embedding(x)
        return x


class LSTM(SequenceModel):
    def __init__(self, num_embeddings: int, embedding_dim: int, num_layers: int,
                 drop_ratio: float, device: str):
        super().__init__(num_embeddings, embedding_dim,
                         pad=True, device=device)
        self.lstm = nn.LSTM(embedding_dim, embedding_dim, num_layers,
                            dropout=drop_ratio, batch_first=True)

        self.init_params()
        assert self.out_embedding.weight.shape == self.in_embedding.weight[:-1].shape
        self.out_embedding.weight = nn.Parameter(self.in_embedding.weight[:-1])
        nn.init.zeros_(self.in_embedding.weight[-1])  # pad index

    def forward(self, x):
        x, lengths = x
        x = x.to(self.device)
        x = self.in_embedding(x)
        x = nn.utils.rnn.pack_padded_sequence(x, lengths, batch_first=True)
        x, _ = self.lstm(x)
        x = self.out_embedding(x.data)
        return x


class BioFixedLenDataset(Dataset):
    def __init__(self, corpus: list, window: int):
        self.window = window

        self.corpus = [np.array(bio, dtype=np.int64) for bio in corpus]
        self.lengths = [len(bio) - window for bio in self.corpus]
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
        x = selected_bio[local_idx:local_idx + self.window]
        y = selected_bio[local_idx + self.window]
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
        lengths = [len(sample) - 1 for sample in batch]
        x = np.full((len(batch), len(batch[0])),
                    fill_value=self.pad_val, dtype=np.int64)
        y = np.full((len(batch), len(batch[0])),
                    fill_value=self.pad_val, dtype=np.int64)
        for i, row in enumerate(batch):
            x[i, :len(row) - 1] = row[:-1]
            y[i, :len(row) - 1] = row[1:]
        y = torch.as_tensor(y)
        y = nn.utils.rnn.pack_padded_sequence(y, lengths, batch_first=True)
        return (torch.as_tensor(x), lengths), y.data


def encode(fnames: list,
           count_threshold: int,
           length_threshold: int,
           vocab: dict = None):
    special_tok = ['<start_bio>', '<end_bio>', '[FAKE]', '[REAL]']
    tokens = []

    # read all files as tokens, cleaning up text data
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
        orders = list(counts.keys())
        orders.sort()
        vocab = {k: Token(k, i, counts[k])
                 for i, k in enumerate(orders)}

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


def save_model(model: nn.Module, path: str):
    if type(model) == FFNN:
        print('Model is FFNN')
        path = os.path.join(path, 'FFNN.pth')
    elif type(model) == LSTM:
        print('Model is LSTM')
        path = os.path.join(path, 'LSTM.pth')
    else:
        raise NotImplementedError
    torch.save(model.state_dict(), path)


def train_categorical(model: nn.Module,
                      optim: torch.optim.Optimizer,
                      criterion: Callable,
                      clip: float,
                      train_loader: DataLoader,
                      valid_loader: DataLoader,
                      epochs: int,
                      path: str,
                      device: str):
    model = model.to(device)
    print('---Training Started---')
    total_step = 0
    train_perplexity_per_epoch, valid_perplexity_per_epoch = [], []
    for epoch in range(1, epochs + 1):
        train_loss = 0.
        t = time.time()
        model.train()
        total_predicted = 0
        for i, (x, y) in tqdm.tqdm(enumerate(train_loader),
                                   total=len(train_loader)):
            total_step += 1
            optim.zero_grad(set_to_none=True)
            pred = model(x)
            y = y.to(device)
            loss = criterion(pred, y)
            temp = loss.detach()
            loss = loss.mean()
            loss.backward()
            nn.utils.clip_grad_value_(model.parameters(), clip)
            optim.step()

            with torch.inference_mode():
                train_loss += temp.sum().cpu()
                total_predicted += y.shape[0]
        with torch.inference_mode():
            model.eval()

            # calculate perplexity for train and valid set
            train_loss = float(train_loss)
            train_loss /= total_predicted
            train_perplexity = np.exp(train_loss)
            train_perplexity_per_epoch.append(train_perplexity)

            valid_loss = 0.
            total_predicted = 0
            for i, (x, y) in enumerate(valid_loader):
                y = y.to(device)
                pred = model(x)
                loss = criterion(pred, y)
                valid_loss += loss.sum().cpu()
                total_predicted += y.shape[0]
            valid_loss = float(valid_loss)
            valid_loss /= total_predicted
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
    with open(os.path.join(path, 'train_results.pkl'), 'wb') as f:
        pickle.dump(results, f)
    save_model(model, path)

    return results


def compute_seq_prob(model, seq, window, vocab, device):
    num_windows = len(seq) - window
    # initialize a tensor of length vocab_size
    log_prob = torch.zeros(len(vocab), device=device)
    for i in range(num_windows):
        x = seq[i:i + window]
        y = model(torch.tensor(x, device=device).unsqueeze(0))
        log_prob += torch.log_softmax(y.squeeze(0), dim=0)
    return log_prob


@torch.inference_mode()
def test_FFNN(model, test_corpus, train_corpus, window, vocab, device):
    # fit a KNN to the training data
    # initialize two np array
    X, y = [], []
    real_label_idx, fake_label_idx = vocab['[REAL]'].idx, vocab['[FAKE]'].idx

    train_corpus = [np.array(bio, dtype=np.int64) for bio in train_corpus]
    print("The length of train_corpus is {}".format(len(train_corpus)))

    true_label_count, false_label_count = 0, 0
    label_count = 100
    iterations = 0
    for seq in train_corpus:
        iterations += 1
        if iterations % 10 == 0:
            print("Trained {} true labels and {} false labels".format(true_label_count, false_label_count))
        append = False
        if true_label_count > label_count and false_label_count > label_count: break
        if seq[-1] == real_label_idx:
            if true_label_count <= label_count:
                append = True
                true_label_count += 1
        else:
            if false_label_count <= label_count:
                append = True
                false_label_count += 1
        if append:
            seq_prob = compute_seq_prob(model, seq, window, vocab, device)
            X.append(seq_prob.detach().cpu().numpy())
            y.append(seq[-1])

    X = np.array(X)
    y = np.array(y)
    knn = KNeighborsClassifier(n_neighbors=1, metric=jensenshannon)
    knn.fit(X, y)
    print('KNN fitted')

    # compute the log probability of each sequence in the test set
    model = model.to(device)
    model.eval()
    test_data = [np.array(bio, dtype=np.int64) for bio in test_corpus]

    TP, FP, FN, TN = 0, 0, 0, 0
    iteration = 0
    for seq in test_data:
        iteration += 1
        if iteration % 10 == 0:
            print("{} of {} iterations".format(iteration, len(test_data)))
        seq_prob = compute_seq_prob(model, seq, window, vocab, device)
        # get the label of the sequence
        label = seq[-1]
        # TODO: solve run time warning in distance calculation: 
        # invalid value encountered in sqrt return np.sqrt(js / 2.0)
        predicted_label = knn.predict(seq_prob.detach().cpu().numpy().reshape(1, -1))
        if predicted_label == vocab['[REAL]'].idx:
            if label == vocab['[REAL]'].idx:
                TP += 1
            else:
                FP += 1
        else:
            if label == vocab['[FAKE]'].idx:
                TN += 1
            else:
                FN += 1
    accuracy = (TP + TN) / (TP + FP + FN + TN)
    results = {
        'accuracy': accuracy,
        'confusion_matrix': np.array([[TP, FP], [FN, TN]]),
    }
    return results


@torch.inference_mode()
def test_LSTM(model, test_corpus, vocab, device):
    model = model.to(device)
    model.eval()
    test_dataset = BioVariableLenDataset(test_corpus, len(vocab))
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn=test_dataset.collate)
    TP, FP, FN, TN = 0, 0, 0, 0
    print('testing started')
    for i, (x, y) in tqdm.tqdm(enumerate(test_loader), total=len(test_loader)):
        y = y.to(device)
        final_token_logits = model(x)[-1]
        label = y[-1]
        if final_token_logits[vocab['[FAKE]'].idx] > final_token_logits[vocab['[REAL]'].idx]:
            pred = vocab['[FAKE]'].idx
        else:
            pred = vocab['[REAL]'].idx
        if pred == label:
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
        'confusion_matrix': np.array([[TP, FP], [FN, TN]]),
    }
    return results


def parse_arguments():
    parser = argparse.ArgumentParser()
    # todo: add more arguments
    parser.add_argument('--model_type', type=str, default='FFNN')
    parser.add_argument('--embedding_dim', type=int, default=128)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--window', type=int, default=5)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--dropout', type=float, default=0.3)
    parser.add_argument('--clip', type=float, default=2.0)
    parser.add_argument('--load_path', type=str, default='')
    parser.add_argument('--weight_decay', type=float, default=0.000025)

    return parser.parse_args()


def plot_learning_curve(train_perplexity: list, valid_perplexity: list,
                        model_type: str, path: str):
    epochs = range(1, len(train_perplexity) + 1)
    plt.plot(epochs, train_perplexity, 'b', label='Train')
    plt.plot(epochs, valid_perplexity, 'r', label='Validation')
    plt.xlabel('Epoch')
    plt.ylabel('Perplexity')
    plt.legend()
    plt.savefig(os.path.join(path, f'perplexity_plot_{model_type}.png'))
    plt.show()


def plot_confusion_matrix(confusion_matrix: np.ndarray, model_type: str, path: str):
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
                           ha="center", va="center", color="w", fontsize=15, fontweight='bold')
    ax.set_title("Confusion Matrix")
    ax.set_ylabel("Predicted")
    ax.set_xlabel("Actual")
    # Save the figure
    plt.savefig(os.path.join(path, f'confusion_matrix_{model_type}.png'))


def main():
    params = parse_arguments()

    (
        model_type,
        embedding_dim,
        num_layers,
        batch_size,
        window,
        epochs,
        lr,
        dropout,
        clip,
        load_path,
        weight_decay
    ) = (
        params.model_type,
        params.embedding_dim,
        params.num_layers,
        params.batch_size,
        params.window,
        params.epochs,
        params.lr,
        params.dropout,
        params.clip,
        params.load_path,
        params.weight_decay
    )

    # device detection
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('using device', device)
    if device != 'cuda':
        print('WARNING: cuda not detected', file=sys.stderr)

    # read tokens, init sequences, dataset, and loader
    vocab, train_corpus = encode(['./mix.train.tok'],
                                 count_threshold=3,
                                 length_threshold=window)
    _, valid_corpus = encode(['./mix.valid.tok'],
                             count_threshold=-1,
                             length_threshold=window,
                             vocab=vocab)
    _, test_corpus = encode(['./mix.test.tok'],
                            count_threshold=-1,
                            length_threshold=window,
                            vocab=vocab)

    for i, corpus in enumerate([train_corpus, valid_corpus, test_corpus]):
        for seq in corpus:
            assert seq[-1].tok in ['[REAL]', '[FAKE]'], f'{i}: {seq[-20:]}'
            assert seq[-2].tok == '<end_bio>', f'{i}: {seq[-20:]}'
            assert seq[0].tok != '<start_bio>', f'{i}: {seq[-20:]}'

    vocab_size = len(vocab)
    print("Vocab size: {}".format(vocab_size))

    # instantiate model and model specific parameters
    if model_type == 'FFNN':
        model = FFNN(num_embeddings=vocab_size,
                     embedding_dim=embedding_dim,
                     window=window,
                     drop_ratio=dropout,
                     device=device)

        train_dataset = BioFixedLenDataset(train_corpus, window)
        valid_dataset = BioFixedLenDataset(valid_corpus, window)
    elif model_type == "LSTM":
        model = LSTM(num_embeddings=vocab_size,
                     embedding_dim=embedding_dim,
                     num_layers=num_layers,
                     drop_ratio=dropout,
                     device=device)

        train_dataset = BioVariableLenDataset(train_corpus, vocab_size)
        valid_dataset = BioVariableLenDataset(valid_corpus, vocab_size)
    else:
        raise NotImplementedError("Neither FFNN nor LSTM")

    criterion = nn.CrossEntropyLoss(reduction='none').to(device)

    train_loader = DataLoader(train_dataset,
                              batch_size=batch_size,
                              shuffle=True,
                              collate_fn=train_dataset.collate)
    print('training dataset loaded with length', len(train_dataset))

    valid_loader = DataLoader(valid_dataset,
                              batch_size=batch_size,
                              shuffle=False,
                              collate_fn=valid_dataset.collate)
    print('validation dataset loaded with length', len(valid_dataset))

    # loading saved params
    if load_path:
        model.load_state_dict(torch.load(os.path.join(load_path, f'{model_type}.pth')))
        path = load_path

    # training a new model
    else:
        path = os.path.join(model_type, f'{int(time.time())}')
        assert not os.path.exists(path)
        os.makedirs(path)
        optim = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

        # start training for categorical prediction
        results = train_categorical(model=model,
                                    optim=optim,
                                    criterion=criterion,
                                    clip=clip,
                                    train_loader=train_loader,
                                    valid_loader=valid_loader,
                                    epochs=epochs,
                                    path=path,
                                    device=device)

        plot_learning_curve(results['train_perplexity'],
                            results['valid_perplexity'],
                            model_type,
                            path)

    if not load_path:
        with open(os.path.join(path, 'arguments.txt'), 'w+') as f:
            f.write(str(params))

    print('---Testing Model---')
    if model_type == "LSTM":
        test_results = test_LSTM(model=model,
                                 test_corpus=test_corpus,
                                 vocab=vocab,
                                 device=device)

    else:
        test_results = test_FFNN(model=model,
                                 test_corpus=test_corpus,
                                 train_corpus=train_corpus,
                                 window=window,
                                 vocab=vocab,
                                 device=device)

    print("The test accuracy is {}".format(test_results['accuracy']))
    plot_confusion_matrix(test_results['confusion_matrix'], model_type, path)


if __name__ == '__main__':
    np.random.seed(0)
    random.seed(0)
    torch.manual_seed(0)
    main()
