import os
import sys
from typing import Callable, Any
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
from torch.nn import LogSoftmax, NLLLoss
from torch.utils.data.dataloader import DataLoader
from collections import defaultdict
from torchtext.vocab import Vocab
from torch.utils.data.dataset import Dataset, TensorDataset
from pathlib import Path
from collections import Counter
from os.path import join as opj
import pathlib
import time
from chu_liu_edmonds import decode_mst
import matplotlib.pyplot as plt
from tqdm import tqdm

curr_dir = pathlib.Path(__file__).parent.absolute()

if curr_dir not in sys.path:
    sys.path.append(curr_dir)

data_dir = opj(curr_dir, '../data')
net_results_dir = opj(curr_dir, '../net_results')
path_train = opj(data_dir, "train.labeled")
path_test = opj(data_dir, "test.labeled")

torch.manual_seed(1)
import dataset, models


DEBUG = 500

def plot_net_results(acc_list, loss_list, epoch, dir_save_path, prefix_str=""):
    fig, ax1 = plt.subplots(figsize=(12, 4))
    ax1.plot(list(range(1, 1 + len(loss_list))), loss_list)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.tick_params(axis='y')

    if len(acc_list) !=0:
        ax2 = ax1.twinx()
        ax2.plot(list(range(len(acc_list))), acc_list)
        ax2 = ax1.twinx()
        ax2.set_ylabel('UAS')
        ax2.tick_params(axis='y')

    fig.tight_layout()
    fig.suptitle(f'Results summary for epoch: {epoch} ')
    fig.savefig(opj(dir_save_path, prefix_str + f"prog_plot_epoch {epoch}"))
    plt.close()

def nll_loss_func(scores, target):
    """

    :param scores: [batch_size, seq_length, seq_length]
    :param target:  [batch_size, seq_length]
    :param nllloss
    :return:
    """
    output= 0
    nllloss = NLLLoss(ignore_index=-1)
    m = LogSoftmax(dim=1)
    output = nllloss(m(scores), target)
    return output

def predict_edges(scores):
    edge_predictions = []
    for sentence_scores in scores:
        score_matrix = sentence_scores.cpu().detach().numpy()
        score_matrix[:, 0] = float("-inf")
        mst, _ = decode_mst(score_matrix, len(score_matrix), has_labels=False)
        edge_predictions.append(mst)
    return np.array(edge_predictions)

def predict(net, device, loader, loss_func):
    net.eval()
    acc, num_of_edges, loss = 0, 0, 0
    for i, sentence in enumerate(tqdm(loader)):
        if DEBUG is not None and i == DEBUG:
            break

        headers = sentence[2].to(device)
        scores = net(sentence)
        loss += loss_func(scores, headers).item()
        predictions = predict_edges(scores)[:, 1:]
        headers = headers.to("cpu").numpy()[:, 1:]
        acc += np.sum(headers == predictions)
        num_of_edges += predictions.size
    net.train()
    return acc / num_of_edges, loss

def train_net(net, train_dataloader, test_dataloader, loss_func: Callable, EPOCHS = 15, BATCH_SIZE = 1, lr=0.001,
              plot_progress=True, results_dir_path='.'):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device used is: {device}")
    net.to(device)
    net.train()
    optimizer = optim.Adam(net.parameters(), lr=lr)

    print("Training Started")
    test_loss_lst, test_acc_lst, train_loss_lst= [], [], []
    best_acc = 0

    for epoch in range(EPOCHS):
        t0 = time.time()
        NUM_WORDS_BATCH = 0
        for i, sentence in enumerate(tqdm(train_dataloader)):
            if DEBUG is not None and i == DEBUG:
                break
            headers = sentence[2].to(device)
            sentence_len = sentence[3].to(device)
            scores = net(sentence)
            loss = loss_func(scores, headers) * sentence_len
            NUM_WORDS_BATCH += sentence_len
            loss.backward()

            if i % BATCH_SIZE == 0:
                loss /= NUM_WORDS_BATCH
                train_loss_lst.append(loss.to('cpu'))
                optimizer.step()
                net.zero_grad()
                NUM_WORDS_BATCH = 0

        test_acc, test_loss = predict(net, device, test_dataloader, loss_func)
        test_loss_lst.append(test_loss)
        test_acc_lst.append(test_acc)

        if best_acc < test_acc and epoch > 5 and test_acc > 0.7:
            save_path = opj(results_dir_path, '_epoch_' + str(epoch) + '_acc_' + str(np.round(test_acc, 4)) + '.pt')
            net.save(save_path)
            best_acc = test_acc

        print(f"Epoch [{epoch + 1}/{EPOCHS}]. \t Test Loss: {test_loss:.2f}"
              f" \t Test Accuracy: {test_acc:.2f}."
              f"Train average Loss: {np.average(train_loss_lst):.2f}"
              f" Time for epoch: {time.time()-t0}")

    plot_net_results(test_acc_lst, test_loss_lst, epoch, results_dir_path, 'test_res_plots')

    if plot_progress:
        plot_net_results([], train_loss_lst, epoch, results_dir_path, 'train_res_plots.png')

##################################################################################################################
if __name__ == '__main__':
    WORD_EMBEDDING_DIM = 300
    TAG_EMBEDDING_DIM = 46
    HIDDEN_DIM = 100
    BATCH_SIZE = 40
    EPOCHS = 15
    LR = 0.01

    paths_list = [path_train, path_test]
    word_dict, pos_dict = dataset.get_vocabs(paths_list)
    train_dataset = dataset.PosDataset(word_dict, pos_dict, data_dir, 'train', padding=False)
    train_dataloader = DataLoader(train_dataset, shuffle=True)

    test_dataset = dataset.PosDataset(word_dict, pos_dict, data_dir, 'test', padding=False)
    test_dataloader = DataLoader(test_dataset, shuffle=False)

    word_vocab_size = len(train_dataset.word_idx_mappings)
    tag_vocab_size = len(train_dataset.pos_idx_mappings)

    base_model = models.BasicDependencyParserModel(word_vocab_size, tag_vocab_size, WORD_EMBEDDING_DIM, TAG_EMBEDDING_DIM,
                                              hidden_dim=HIDDEN_DIM, mlp_dim_out=32)

    res_dir = opj(net_results_dir,time.strftime("%Y%m%d-%H%M%S") )
    if not res_dir:
        os.makedirs(res_dir)
    train_net(base_model, train_dataloader, test_dataloader, nll_loss_func, EPOCHS=EPOCHS, BATCH_SIZE=BATCH_SIZE, lr=LR,
              plot_progress=True, results_dir_path= res_dir)



