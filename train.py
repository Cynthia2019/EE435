import os
import time
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


class Token:
    __slots__ = ['tok', 'idx', 'count']

    def __init__(self, tok, idx, count=0):
        self.tok = tok
        self.idx = idx
        self.count = count

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return self.tok == other.tok and self.idx == other.idx and self.count == other.count
        else:
            return self.tok == other

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
        self.dropout1 = nn.Dropout(drop_ratio) if drop_ratio else None
        self.dropout2 = nn.Dropout(drop_ratio) if drop_ratio else None

        self.init_params()
        self.out_embedding.weight = self.in_embedding.weight

    def forward(self, x: torch.Tensor):
        x = x.to(self.device)
        x = self.in_embedding(x)
        x = torch.flatten(x, 1)
        x = self.dropout1(x) if self.dropout1 else x
        x = self.linear(x)
        x = torch.tanh(x)
        x = self.dropout2(x) if self.dropout2 else x
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
        self.blind_mode = corpus[0][-1].tok not in ['[REAL]', '[FAKE]']
        if self.blind_mode:
            for bio in corpus:
                assert bio[-1].tok not in ['[REAL]', '[FAKE]']
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
        lengths = [len(sample) - (1 - self.blind_mode) for sample in batch]
        x = np.full((len(batch), lengths[0]),
                    fill_value=self.pad_val, dtype=np.int64)
        y = np.full((len(batch), len(batch[0]) - 1),
                    fill_value=self.pad_val, dtype=np.int64)
        for i, (sample, length) in enumerate(zip(batch, lengths)):
            x[i, :length] = sample[:length]
            y[i, :len(sample) - 1] = sample[1:]
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
    corpus = [seq for i, seq in enumerate(corpus) if seq not in corpus[:i]]

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
    print('---Training Started---')
    total_step = 0
    train_perplexity_per_epoch, valid_perplexity_per_epoch = [], []
    for epoch in range(1, epochs + 1):
        train_loss = 0.
        t = time.time()
        model.train()
        total_predicted = 0
        for i, (x, y) in enumerate(train_loader):
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
    save_model(model, path)

    return results


def get_confusion_matrix(preds, labels, positive_idx):
    preds, labels = np.array(preds, dtype=np.int64), np.array(labels, dtype=np.int64)
    true = preds == labels
    false = ~true
    positive = preds == positive_idx
    negative = ~positive
    TP = (true & positive).sum()
    FP = (false & positive).sum()
    FN = (false & negative).sum()
    TN = (true & negative).sum()
    return np.array([[TP, FP], [FN, TN]])


def torch_js(p, q):  # js divergence
    def kl_div(a, b):  # kl divergence
        result = a * (a / b).log_()
        return result.nansum(dim=1)

    m = p + q
    m *= 0.5
    out = kl_div(p, m) + kl_div(q, m)
    out *= 0.5
    return out.sqrt_()


@torch.inference_mode()
def test_FFNN_KNN(model, test_corpus, train_corpus, window, vocab, metric, device):
    """
    test FFNN by using KNN as classifier.

    steps:
    1. for each bio in training and test set, do:
        1) get the probability distribution over vocab
            for each window of the sequence using FFNN
        2) compute the average of all probability distributions from the same sequence
        3) normalize the averaged probability distribution
    2. for each probability distribution in test set, do:
        use JS divergence/l1/l2 as metric against train set
        classify using K nearest neighbors
    3. compute confusion matrix and accuracy
    """
    model.eval()
    criterion = nn.CrossEntropyLoss(reduction='sum').to(device)

    def get_sequence_dist(bio_seq):
        batch = [bio_seq[i:i + window] for i in range(len(bio_seq) - window)]
        target = bio_seq[window:]
        batch = np.array(batch, dtype=np.int64)
        batch = torch.as_tensor(batch, device=device)
        target = torch.as_tensor(target, device=device)
        pred = model(batch)
        batch_loss = criterion(pred, target)
        pred_probs = torch.softmax(pred, dim=1)
        avg_pred = pred_probs.mean(dim=0)
        avg_pred /= avg_pred.sum()  # normalize
        return avg_pred.cpu().numpy(), batch_loss, len(target)

    train_dist, train_labels = [], []
    for bio in train_corpus:
        train_labels.append(bio[-1].idx)
        bio = bio[:-1]
        bio = np.array(bio, dtype=np.int64)
        train_dist.append(get_sequence_dist(bio)[0])

    train_dist = np.array(train_dist, dtype=np.float32)
    train_labels = np.array(train_labels, dtype=np.int64)
    train_dist = torch.as_tensor(train_dist, device=device)
    train_labels = torch.as_tensor(train_labels, device=device)

    total_loss, total_predicted = 0., 0
    test_dist, test_labels = [], []
    for bio in test_corpus:
        if bio[-1].tok in ['[REAL]', '[FAKE]']:
            test_labels.append(bio[-1].idx)
            bio = bio[:-1]
        bio = np.array(bio, dtype=np.int64)
        distribution, loss, count = get_sequence_dist(bio)
        test_dist.append(distribution)
        total_loss += loss
        total_predicted += count
    total_loss /= total_predicted
    perplexity = torch.exp(total_loss).cpu().numpy()

    print('starting KNN prediction')
    if metric == 'l2':
        metric = lambda x, y: torch.linalg.norm(x - y, dim=1)
    elif metric == 'l1':
        metric = lambda x, y: torch.linalg.norm(x - y, ord=1, dim=1)
    elif metric == 'js':
        metric = torch_js
    else:
        raise NotImplementedError(metric)
    test_dist = np.array(test_dist, dtype=np.float32)
    test_dist = torch.as_tensor(test_dist, device=device)
    test_preds = []
    for prob_distribution in test_dist:
        distance_vec = metric(prob_distribution.unsqueeze(0), train_dist)
        topk = torch.topk(distance_vec, k=65, sorted=False, largest=False).indices
        nearest = train_labels[topk]
        pred_label = torch.mode(nearest).values.cpu().numpy()
        test_preds.append(pred_label)
    results = {}
    if test_labels:
        conf_matrix = get_confusion_matrix(test_preds, test_labels, vocab['[REAL]'].idx)
        accuracy = (conf_matrix[0, 0] + conf_matrix[1, 1]) / conf_matrix.sum()
        results = {
            'accuracy': accuracy,
            'confusion_matrix': conf_matrix
        }
    results['test_perplexity'] = perplexity
    results['test_predictions'] = test_preds
    return results


@torch.inference_mode()
def test_FFNN_chain(model, test_corpus, train_corpus, window, vocab, path, device):
    """
    classify FFNN using chain rule as described during lecture,
    note that different from KNN method,
    KNN uses average distribution to represent a sequence (a vector),
    this uses average cross entropy  to represent a sequence (a scaler)

    However, instead of manually estimating the intersection from histogram plot,
    we compute the approximate intersection by looking for overlap of two histograms
    """
    model.eval()
    criterion = nn.CrossEntropyLoss(reduction='sum').to(device)

    def get_sequence_prob(bio_seq):
        batch = [bio_seq[i:i + window] for i in range(len(bio_seq) - window)]
        target = bio_seq[window:]
        target = torch.as_tensor(target, device=device)
        assert len(batch) == len(target)
        batch = np.array(batch, dtype=np.int64)
        batch = torch.as_tensor(batch, device=device)
        pred = model(batch)
        batch_loss = criterion(pred, target)
        return (batch_loss / len(target)).cpu().numpy(), batch_loss, len(target)

    seq_probs = {'[REAL]': [], '[FAKE]': []}
    for bio in train_corpus:
        label = bio[-1].tok
        bio = bio[:-1]
        bio = np.array(bio, dtype=np.int64)
        seq_probs[label].append(get_sequence_prob(bio)[0])
    for k in seq_probs.keys():
        seq_probs[k] = np.array(seq_probs[k])

    print('finding intersection')
    lowest = min(np.min(seq_probs['[REAL]']), np.min(seq_probs['[FAKE]']))
    highest = max(np.max(seq_probs['[REAL]']), np.max(seq_probs['[FAKE]']))
    bin_range = (lowest - 1e-7, highest + 1e-7)
    real_hist, bins = np.histogram(seq_probs['[REAL]'], 200, range=bin_range)
    fake_hist, _ = np.histogram(seq_probs['[FAKE]'], bins)
    real_hist_smoothed = np.array([real_hist[max(0, i - 2): i + 2].mean()
                                   for i in range(len(real_hist))])
    fake_hist_smoothed = np.array([fake_hist[max(0, i - 2): i + 2].mean()
                                   for i in range(len(fake_hist))])
    overlap_hist = np.minimum(real_hist_smoothed, fake_hist_smoothed)
    real_median = np.median(seq_probs['[REAL]'])
    intersect_range = [real_median,
                       np.median(seq_probs['[FAKE]'])]
    lower, upper = sorted(intersect_range)
    lower = np.argwhere(bins >= lower)[0, 0]
    upper = np.argwhere(bins >= upper)[0, 0]
    mask = np.ones_like(overlap_hist)
    mask = mask.astype(bool)
    mask[lower:upper] = False
    overlap_hist = np.ma.masked_array(overlap_hist, mask)
    intersection = np.argmax(overlap_hist)
    bins_center = (bins[1:] + bins[:-1]) / 2.
    intersection = bins_center[intersection]
    print('estimated decision boundary is', intersection)
    if path is not None:
        plt.plot(bins_center, real_hist, c='b', label='Real')
        plt.plot(bins_center, fake_hist, c='r', label='Fake')
        plt.axvline(x=intersection, color='k', label='Decision Boundary')
        plt.xlabel('-Avg. Log Prob')
        plt.ylabel('Frequency')
        plt.legend()
        plt.savefig(os.path.join(path, 'histogram_plot_FFNN.png'))

    real_idx = vocab['[REAL]'].idx
    fake_idx = vocab['[FAKE]'].idx
    total_loss, total_predicted = 0., 0
    test_preds, test_labels = [], []
    for bio in test_corpus:
        if bio[-1].tok in ['[REAL]', '[FAKE]']:
            test_labels.append(bio[-1].idx)
            bio = bio[:-1]
        bio = np.array(bio, dtype=np.int64)
        seq_prob, loss, count = get_sequence_prob(bio)
        total_loss += loss
        total_predicted += count

        if (real_median < intersection) == (seq_prob < intersection):
            test_preds.append(real_idx)
        else:
            test_preds.append(fake_idx)
    total_loss /= total_predicted
    perplexity = torch.exp(total_loss).cpu().numpy()

    results = {}
    if test_labels:
        conf_matrix = get_confusion_matrix(test_preds, test_labels, real_idx)
        accuracy = (conf_matrix[0, 0] + conf_matrix[1, 1]) / conf_matrix.sum()
        results = {
            'accuracy': accuracy,
            'confusion_matrix': conf_matrix,
        }
    results['test_perplexity'] = perplexity
    results['test_predictions'] = test_preds
    return results


@torch.inference_mode()
def test_LSTM(model, test_corpus, vocab, device):
    model.eval()
    criterion = nn.CrossEntropyLoss(reduction='sum').to(device)
    test_dataset = BioVariableLenDataset(test_corpus, len(vocab))
    if test_dataset.blind_mode:
        print('blind mode on')
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False,
                             collate_fn=test_dataset.collate)
    preds, labels = [], []
    real_idx = vocab['[REAL]'].idx
    fake_idx = vocab['[FAKE]'].idx
    print('testing started')
    total_loss, total_predicted = 0., 0
    for i, (x, y) in enumerate(test_loader):

        total_predicted += len(y)
        pred = model(x)
        final_token_logits = pred[-1].cpu().numpy()
        if test_dataset.blind_mode:
            pred = pred[:-1]
        else:
            labels.append(y[-1])
        assert len(pred) == len(y)
        total_loss += criterion(pred, y.to(device))
        pred = final_token_logits[fake_idx] > final_token_logits[real_idx]
        pred = fake_idx if pred else real_idx
        preds.append(pred)
    total_loss /= total_predicted
    perplexity = torch.exp(total_loss).cpu().numpy()
    results = {}
    if labels:
        conf_matrix = get_confusion_matrix(preds, labels, real_idx)
        accuracy = (conf_matrix[0, 0] + conf_matrix[1, 1]) / conf_matrix.sum()
        results = {
            'accuracy': accuracy,
            'confusion_matrix': conf_matrix,
        }
    results['test_perplexity'] = perplexity
    results['test_predictions'] = preds
    return results


@torch.inference_mode()
def ensemble(model, test_corpus, train_corpus, window, vocab, device):
    print('KNN l1...')
    test_results1 = test_FFNN_KNN(model=model,
                                  test_corpus=test_corpus,
                                  train_corpus=train_corpus,
                                  window=window,
                                  vocab=vocab,
                                  metric='l1',
                                  device=device)
    print('KNN l2...')
    test_results2 = test_FFNN_KNN(model=model,
                                  test_corpus=test_corpus,
                                  train_corpus=train_corpus,
                                  window=window,
                                  vocab=vocab,
                                  metric='l2',
                                  device=device)
    print('KNN js...')
    test_results3 = test_FFNN_KNN(model=model,
                                  test_corpus=test_corpus,
                                  train_corpus=train_corpus,
                                  window=window,
                                  vocab=vocab,
                                  metric='js',
                                  device=device)
    test_labels = [bio[-1] for bio in test_corpus if bio[-1].tok in ['[REAL]', '[FAKE]']]
    test_preds = [test_results1['test_predictions'],
                  test_results2['test_predictions'],
                  test_results3['test_predictions']]
    # majority voting
    test_preds = [round(sum(preds) / len(preds)) for preds in zip(*test_preds)]
    results = {}
    if test_labels:
        conf_matrix = get_confusion_matrix(test_preds, test_labels, vocab['[REAL]'].idx)
        accuracy = (conf_matrix[0, 0] + conf_matrix[1, 1]) / conf_matrix.sum()
        results = {
            'accuracy': accuracy,
            'confusion_matrix': conf_matrix,
        }
    results['test_perplexity'] = test_results1['test_perplexity']
    results['test_predictions'] = test_preds
    return results


def plot_learning_curve(train_perplexity: list, valid_perplexity: list,
                        model_type: str, path: str):
    epochs = range(1, len(train_perplexity) + 1)
    plt.plot(epochs, train_perplexity, 'b', label='Train')
    plt.plot(epochs, valid_perplexity, 'r', label='Validation')
    plt.xticks(np.arange(1, len(train_perplexity) + 1, len(train_perplexity) // 5))
    plt.xlabel('Epoch')
    plt.ylabel('Perplexity')
    plt.legend()
    plt.savefig(os.path.join(path, f'perplexity_plot_{model_type}.png'))
    # plt.show()


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


def seed_all(device):
    if device == 'cuda':
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    os.environ['PYTHONHASHSEED'] = '1'
    random.seed(1)
    np.random.seed(1)
    torch.manual_seed(1)
    if device == 'cuda':
        from torch.backends import cudnn
        torch.cuda.manual_seed(1)
        torch.cuda.manual_seed_all(1)
        cudnn.benchmark = False
        cudnn.deterministic = True
    torch.set_deterministic_debug_mode('warn')
    torch.use_deterministic_algorithms(True, warn_only=True)


def save_pred(predictions, vocab, path):
    vocab = list(vocab.values())
    predictions = [vocab[pred].tok for pred in predictions]
    for pred in predictions:
        assert pred in ['[REAL]', '[FAKE]']

    with open(os.path.join(path, 'blind_predictions.csv'),
              'w+', encoding='utf8') as f:
        f.write('\n'.join(predictions))


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', type=str, default='FFNN', choices=['FFNN', 'LSTM'])
    parser.add_argument('--embedding_dim', type=int, default=512)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--window', type=int, default=5)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--dropout', type=float, default=0.3)
    parser.add_argument('--clip', type=float, default=2.0)
    parser.add_argument('--load_path', type=str, default='')
    parser.add_argument('--weight_decay', type=float, default=0.000025)
    parser.add_argument('--classifier', type=str, default='KNN_js',
                        choices=['KNN_l1', 'KNN_l2', 'KNN_js', 'ensemble', 'chain'])

    return parser.parse_args()


def main():
    # device detection
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    seed_all(device)
    params = parse_arguments()
    print(params)

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
        weight_decay,
        classifier
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
        params.weight_decay,
        params.classifier
    )

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

        if not load_path:
            train_dataset = BioFixedLenDataset(train_corpus, window)
            valid_dataset = BioFixedLenDataset(valid_corpus, window)
    elif model_type == "LSTM":
        model = LSTM(num_embeddings=vocab_size,
                     embedding_dim=embedding_dim,
                     num_layers=num_layers,
                     drop_ratio=dropout,
                     device=device)

        if not load_path:
            train_dataset = BioVariableLenDataset(train_corpus, vocab_size)
            valid_dataset = BioVariableLenDataset(valid_corpus, vocab_size)
    else:
        raise NotImplementedError("Neither FFNN nor LSTM")

    criterion = nn.CrossEntropyLoss(reduction='none').to(device)
    model = model.to(device)
    if device == 'cuda':
        torch.cuda.synchronize()

    path = os.path.join(model_type, f'{int(time.time())}{"test" if load_path else ""}')
    assert not os.path.exists(path)
    os.makedirs(path)
    print(f'saving to {path}')

    # loading saved params
    if load_path:
        print(f'loaded from {load_path}')
        model.load_state_dict(torch.load(os.path.join(load_path, f'{model_type}.pth'),
                                         map_location=device))
        with open(os.path.join(load_path, f'{model_type}_train_results.pkl'), 'rb') as f:
            results = pickle.load(f)

    # training a new model
    else:
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
        with open(os.path.join(path, f'{model_type}_train_results.pkl'), 'wb') as f:
            pickle.dump(results, f)

    plot_learning_curve(results['train_perplexity'],
                        results['valid_perplexity'],
                        model_type,
                        path)

    with open(os.path.join(path, 'arguments.txt'), 'w+') as f:
        f.write(str(params))

    print('---Testing Model---')
    _, blind_corpus = encode(['./blind.test.tok'],
                             count_threshold=-1,
                             length_threshold=1,
                             vocab=vocab)

    t = time.time()
    if model_type == "LSTM":
        test_results = test_LSTM(model=model,
                                 test_corpus=test_corpus,
                                 vocab=vocab,
                                 device=device)
        blind_results = test_LSTM(model=model,
                                  test_corpus=blind_corpus,
                                  vocab=vocab,
                                  device=device)


    else:
        if 'KNN' in classifier:
            test_results = test_FFNN_KNN(model=model,
                                         test_corpus=test_corpus,
                                         train_corpus=train_corpus,
                                         window=window,
                                         vocab=vocab,
                                         metric=classifier[4:],
                                         device=device)
            blind_results = test_FFNN_KNN(model=model,
                                          test_corpus=blind_corpus,
                                          train_corpus=train_corpus,
                                          window=window,
                                          vocab=vocab,
                                          metric=classifier[4:],
                                          device=device)
        elif classifier == 'chain':
            test_results = test_FFNN_chain(model=model,
                                           test_corpus=test_corpus,
                                           train_corpus=train_corpus,
                                           window=window,
                                           vocab=vocab,
                                           path=path,
                                           device=device)
            blind_results = test_FFNN_chain(model=model,
                                            test_corpus=blind_corpus,
                                            train_corpus=train_corpus,
                                            window=window,
                                            vocab=vocab,
                                            path=None,
                                            device=device)
        elif classifier == 'ensemble':
            test_results = ensemble(model=model,
                                    test_corpus=test_corpus,
                                    train_corpus=train_corpus,
                                    window=window,
                                    vocab=vocab,
                                    device=device)
            blind_results = ensemble(model=model,
                                     test_corpus=blind_corpus,
                                     train_corpus=train_corpus,
                                     window=window,
                                     vocab=vocab,
                                     device=device)
        else:
            raise NotImplementedError
    assert len(blind_results['test_predictions']) == len(blind_corpus)
    print('blind test length:', len(blind_corpus))
    print(f'test accuracy {test_results["accuracy"]},',
          f'test perplexity {test_results["test_perplexity"]},',
          f'blind test perplexity {blind_results["test_perplexity"]},',
          f'time {time.time() - t}')
    save_pred(blind_results['test_predictions'], vocab, path)
    plot_confusion_matrix(test_results['confusion_matrix'], model_type, path)


if __name__ == '__main__':
    main()
