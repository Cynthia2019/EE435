import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import math
import time
import numpy as np
import sys
import argparse
import os


def decode(vocab, corpus):
    text = ''
    for i in range(len(corpus)):
        wID = corpus[i]
        text = text + vocab[wID] + ' '
    return (text)


def encode(words, text):
    corpus = []
    tokens = text.split(' ')
    for t in tokens:
        try:
            wID = words[t][0]
        except:
            wID = words['<unk>'][0]
        corpus.append(wID)
    return (corpus)


def read_encode(file_name, vocab, words, corpus, threshold):
    wID = len(vocab)
    if threshold > -1:
        with open(file_name, 'rt') as f:
            for line in f:
                line = line.replace('\n', '')
                tokens = line.split(' ')
                for t in tokens:
                    try:
                        elem = words[t]
                    except:
                        elem = [wID, 0]
                        vocab.append(t)
                        wID = wID + 1
                    # elem: id, occurrence
                    elem[1] = elem[1] + 1
                    # words: token -> [Wid, occurrence]
                    words[t] = elem

        # cleaning up words/vocab with respect to threshold
        temp = words
        words = {}
        vocab = []
        wID = 0
        words['<unk>'] = [wID, 100]
        vocab.append('<unk>')
        for t in temp:
            if temp[t][1] >= threshold:
                vocab.append(t)
                wID = wID + 1
                words[t] = [wID, temp[t][1]]

    # add to corpus
    with open(file_name, 'rt') as f:
        for line in f:
            line = line.replace('\n', '')
            tokens = line.split(' ')
            for t in tokens:
                try:
                    wID = words[t][0]
                except:
                    wID = words['<unk>'][0]
                corpus.append(wID)

    return [vocab, words, corpus]


class FFNN(nn.Module):
    def __init__(self, vocab, words, d_model, d_hidden, dropout):
        super().__init__()

        self.vocab = vocab
        self.words = words
        self.vocab_size = len(self.vocab)
        self.d_model = d_model
        self.d_hidden = d_hidden
        self.dropout = dropout
        self.embeds = nn.Embedding(vocab_size, d_model)

    #          {perform other initializations needed for the FFNN}

    def forward(self, src):
        embeds = self.dropout(self.embeds(src))
        #          {add code to implement the FFNN}
        return x

    def init_weights(self):
        #          {perform initializations}
        pass


class LSTM(nn.Module):
    def __init__(self, vocab, words, d_model, d_hidden, n_layers, dropout_rate):
        super().__init__()

        self.vocab = vocab
        self.words = words
        self.vocab_size = len(self.vocab)
        self.n_layers = n_layers
        self.d_hidden = d_hidden
        self.d_model = d_model
        self.embeds = nn.Embedding(self.vocab_size, d_model)

    #          {perform other initializations needed for the LSTM}

    def forward(self, src, h):
        embeds = self.dropout(self.embeds(src))
        #          {add code to implement the LSTM}
        return [preds, h]

    def init_weights(self):
        #          {perform initializations}
        pass

    def detach_hidden(self, hidden):
        #          {needed for training...}
        return [hidden, cell]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d_model', type=int, default=100)
    parser.add_argument('-d_hidden', type=int, default=100)
    parser.add_argument('-n_layers', type=int, default=2)
    parser.add_argument('-batch_size', type=int, default=20)
    parser.add_argument('-seq_len', type=int, default=30)
    parser.add_argument('-printevery', type=int, default=5000)
    parser.add_argument('-window', type=int, default=3)
    parser.add_argument('-epochs', type=int, default=20)
    parser.add_argument('-lr', type=float, default=0.0001)
    parser.add_argument('-dropout', type=int, default=0.35)
    parser.add_argument('-clip', type=int, default=2.0)
    parser.add_argument('-model', type=str, default='LSTM')
    parser.add_argument('-savename', type=str, default='lstm')
    parser.add_argument('-loadname', type=str)
    parser.add_argument('-trainname', type=str, default='real.train.txt')
    parser.add_argument('-validname', type=str, default='real.valid.txt')
    parser.add_argument('-testname', type=str, default='real.test.txt')

    params = parser.parse_args()
    torch.manual_seed(0)

    [vocab, words, train] = read_encode(params.trainname, [], {}, [], 3)
    print('vocab: %d train: %d' % (len(vocab), len(train)))
    [vocab, words, test] = read_encode(params.testname, vocab, words, [], -1)
    print('vocab: %d test: %d' % (len(vocab), len(test)))
    params.vocab_size = len(vocab)

    if params.model == 'FFNN':
        print(params.model)
        model = FFNN(vocab, words, params.d_model, params.d_hidden, params.dropout)
    #          {add code to instantiate the model, train for K epochs and save model to disk}

    if params.model == 'LSTM':
        print(params.model)
        model = LSTM(vocab, words, params.d_model, params.d_hidden, params.n_layers, params.dropout)

    #          {add code to instantiate the model, train for K epochs and save model to disk}

    if params.model == 'FFNN_CLASSIFY':
        print(params.model)
    #          {add code to instantiate the model, recall model parameters and perform/learn classification}

    if params.model == 'LSTM_CLASSIFY':
        print(params.model)
    #          {add code to instantiate the model, recall model parameters and perform/learn classification}

    print(params)
    # tokenized texts
    print(train[0:100])


if __name__ == "__main__":
    main()
