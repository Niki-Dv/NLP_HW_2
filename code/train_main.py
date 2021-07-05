import os
import sys
from itertools import product
from typing import Callable, Any
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
import pickle
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
path_comp = opj(data_dir, "comp.unlabeled")

torch.manual_seed(1)
import dataset, models

ROUND_NUM_DIGITS = 4
DEBUG = None
##################################################################################################################
def plot_net_results(acc_list, loss_list, epoch, dir_save_path, prefix_str=""):

    f = plt.figure()
    axes = f.add_subplot(121)
    axes.plot(list(range(1, 1 + len(loss_list))), loss_list, color='red', label='loss')
    axes.set_xlabel('Epoch')
    axes.set_ylabel('Loss')
    axes.set_title("loss graph")
    axes.legend()

    if len(acc_list) != 0:
        axes = f.add_subplot(122)
        axes.set_title("accuracy graph")
        axes.plot(list(range(len(acc_list))), acc_list, color='blue', label='acc')
        axes.set_ylabel('UAS')
        axes.tick_params(axis='y')
        axes.legend()
    f.tight_layout()
    if len(acc_list) != 0:
        f.savefig(opj(dir_save_path, prefix_str + f"final acc_{round(acc_list[-1],3)}_prog_plot_epoch {epoch}.png"))
    else:
        f.savefig(opj(dir_save_path, prefix_str + f"final loss_{round(loss_list[-1], 3)}_prog_plot_epoch {epoch}.png"))
    plt.close('all')

##################################################################################################################
def nll_loss_func(scores, target):
    """

    :param scores: [batch_size, seq_length, seq_length]
    :param target:  [batch_size, seq_length]
    :param nllloss
    :return:
    """
    nllloss = NLLLoss(ignore_index=-1)
    m = LogSoftmax(dim=1)
    output = nllloss(m(scores), target)
    return output

##################################################################################################################
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
    return acc / num_of_edges, loss / len(loader)
##################################################################################################################
def train_net(net, train_dataloader, test_dataloader, loss_func: Callable, EPOCHS = 15, BATCH_SIZE = 1, lr=0.001,
              plot_progress=True, results_dir_path='.', change_lr=False, consider_sentence_len=False):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device used is: {device}")
    net.to(device)
    net.train()
    optimizer = optim.Adam(net.parameters(), lr=lr)

    print("Training Started")
    test_loss_lst, test_acc_lst, train_loss_lst= [], [], []
    best_acc = 0
    best_loss = 1000
    for epoch in range(EPOCHS):
        if epoch == 8:
            net.word_embedding.weight.requires_grad = True
        if change_lr and epoch >0 and epoch % 2 == 0:
            for g in optimizer.param_groups:
                g['lr'] *= 1e-1

        t0 = time.time()
        total_loss = 0
        num_words_in_batch = 0
        for i, sentence in enumerate(tqdm(train_dataloader)):
            if DEBUG is not None and i == DEBUG:
                break
            headers = sentence[2].to(device)
            num_words_in_batch += sentence[3].item()
            scores = net(sentence)
            loss = loss_func(scores, headers)
            
            total_loss += loss.item()
            if consider_sentence_len:
                loss *= sentence[3].to(device).item()

            loss= loss/BATCH_SIZE
            loss.backward()

            if i % BATCH_SIZE == 0 and i > 0:
                num_words_in_batch = 0
                optimizer.step()
                net.zero_grad()
        train_loss_lst.append(total_loss/len(train_dataloader))
        test_acc, test_loss = predict(net, device, test_dataloader, loss_func)
        test_loss_lst.append(test_loss)
        test_acc_lst.append(test_acc)

        if best_loss > train_loss_lst[-1] and best_acc < test_acc and epoch > 3 and test_acc > 0.7:
            save_path = opj(results_dir_path, '_epoch_' + str(epoch) + '_acc_' + str(np.round(test_acc, 4)) + '.pt')
            net.save(save_path)
            best_acc = test_acc
            print("saved new module")

        print(f"\nEpoch [{epoch + 1}/{EPOCHS}]. \t Test sentence avg loss: {test_loss:.{ROUND_NUM_DIGITS}f}"
              f" \t Test Accuracy: {test_acc:.{ROUND_NUM_DIGITS}f}"
              f"\t Train sentence avg loss: {train_loss_lst[-1]:.{ROUND_NUM_DIGITS}f}"
              f"\t Time for epoch: {time.time()-t0}")

    plot_net_results(test_acc_lst, test_loss_lst, epoch, results_dir_path, 'test_res_plots')

    if plot_progress:
        plot_net_results([], train_loss_lst, epoch, results_dir_path, 'train_res_plots')

    return np.max(test_acc_lst)
##################################################################################################################
def run_base_model():
    WORD_EMBEDDING_DIM = 100
    TAG_EMBEDDING_DIM = 25
    LSTM_HIDDEN_DIM = 125
    MLP_HIDDEN_DIM = 100
    BATCH_SIZE = 50
    EPOCHS = 20
    LR = 0.01

    paths_list = [path_train]
    word_dict, pos_dict = dataset.get_vocabs(paths_list)
    train_dataset = dataset.PosDataset(word_dict, pos_dict, data_dir, 'train', padding=False)
    train_dataloader = DataLoader(train_dataset, shuffle=True)

    test_dataset = dataset.PosDataset(word_dict, pos_dict, data_dir, 'test', padding=False, word_embeddings=None,
                                      alpha_dropout=0.0)
    test_dataloader = DataLoader(test_dataset, shuffle=False)

    word_vocab_size = len(train_dataset.word_idx_mappings)
    tag_vocab_size = len(train_dataset.pos_idx_mappings)

    base_model = models.BasicDependencyParserModel(word_vocab_size, tag_vocab_size, WORD_EMBEDDING_DIM,
                                                   TAG_EMBEDDING_DIM,
                                                   hidden_dim=LSTM_HIDDEN_DIM, mlp_dim_out=MLP_HIDDEN_DIM)

    res_dir = opj(net_results_dir,  "base_model_results" + time.strftime("%Y%m%d-%H%M%S"))
    if not os.path.isdir(res_dir):
        os.makedirs(res_dir)

    train_net(base_model, train_dataloader, test_dataloader, nll_loss_func, EPOCHS=EPOCHS, BATCH_SIZE=BATCH_SIZE, lr=LR,
              plot_progress=True, results_dir_path=res_dir)

##################################################################################################################
def score_to_mst(scores):
    score_matrix = scores.cpu().detach().numpy()
    score_matrix[:, 0] = float("-inf")
    mst, _ = decode_mst(score_matrix, len(score_matrix), has_labels=False)
    return mst


def predict_from_loader(net, loader):
    #doesn't work with batchs only single sentence
    net.eval()
    predictions = []
    for i, sentence in enumerate(loader):
        scores = net(sentence)
        prediction = score_to_mst(scores[0,...])[1:]
        predictions.append(prediction)
    return predictions

def write_comp(path_file_r,path_file_w,predictions):
    sentence_index, word_index = 0, 0
    with open(path_file_r, 'r') as r_file:
        with open(path_file_w, 'w+') as w_file:
            for line in r_file:
                if line == '\n':
                    sentence_index += 1
                    word_index = 0
                    w_file.write(line)
                    continue
                splitted_line = line.split('\t')
                splitted_line[6] = str(predictions[sentence_index][word_index])
                line = '\t'.join(splitted_line)
                w_file.write(line)
                word_index += 1

##################################################################################################################
def run_adv_model():
    WORD_EMBEDDING_DIM = 50
    TAG_EMBEDDING_DIM = 30
    LSTM_HIDDEN_DIM = 125
    MLP_HIDDEN_DIM = 100
    BATCH_SIZE = 20
    EPOCHS = 20
    LR = 0.01
    CHANGE_LR=True

    paths_list = [path_train,path_comp,path_test]
    word_dict, pos_dict = dataset.get_vocabs(paths_list)
    train_dataset = dataset.PosDataset(word_dict, pos_dict, data_dir, 'train', padding=False,
                                       alpha_dropout=0.0, WORD_EMBD_DIM=WORD_EMBEDDING_DIM)
    train_dataloader = DataLoader(train_dataset, shuffle=True)

    word_embd = train_dataset.word_idx_mappings, train_dataset.idx_word_mappings, train_dataset.word_vectors
    test_dataset = dataset.PosDataset(word_dict, pos_dict, data_dir, 'test', padding=False, word_embeddings=word_embd,
                                      alpha_dropout=0.0, WORD_EMBD_DIM=WORD_EMBEDDING_DIM)
    test_dataloader = DataLoader(test_dataset, shuffle=False)

    word_vocab_size = len(train_dataset.word_idx_mappings)
    tag_vocab_size = len(train_dataset.pos_idx_mappings)

    base_model = models.AdvDependencyParserModel(train_dataset.word_vectors, tag_vocab_size, WORD_EMBEDDING_DIM,
                                                   TAG_EMBEDDING_DIM, hidden_dim=LSTM_HIDDEN_DIM,
                                                   mlp_dim_out=MLP_HIDDEN_DIM)

    res_dir = opj(net_results_dir, "base_with_prep_emb_model_results" + time.strftime("%Y%m%d-%H%M%S"))

    if not os.path.isdir(res_dir):
        os.makedirs(res_dir)

    train_net(base_model, train_dataloader, test_dataloader, nll_loss_func, EPOCHS=EPOCHS, BATCH_SIZE=BATCH_SIZE, lr=LR,
              plot_progress=True, results_dir_path=res_dir, change_lr=CHANGE_LR, consider_sentence_len=True)

def run_different_combos():
    WORD_EMBEDDING_DIM_OPTIONS = [50, 100, 200]
    TAG_EMBEDDING_DIM_OPTIONS = [25, 50]
    LSTM_HIDDEN_DIM_OPTIONS = [60, 125, 250]
    MLP_HIDDEN_DIM_OPTIONS = [50, 100, 200]
    BATCH_SIZE_OPTIONS = [40]
    EPOCHS_OPTIONS = [15]
    LR_OPTIONS = [0.01, 0.05]
    change_lr_options = [True, False]

    results_dict = {}
    for combo in product(WORD_EMBEDDING_DIM_OPTIONS, TAG_EMBEDDING_DIM_OPTIONS, LSTM_HIDDEN_DIM_OPTIONS,
                         MLP_HIDDEN_DIM_OPTIONS, BATCH_SIZE_OPTIONS, EPOCHS_OPTIONS, LR_OPTIONS, change_lr_options):
        print(f"Running combo: {combo}")
        WORD_EMBEDDING_DIM, TAG_EMBEDDING_DIM, LSTM_HIDDEN_DIM, MLP_HIDDEN_DIM, BATCH_SIZE, EPOCHS, LR, CHANGE_LR =\
            combo[0], combo[1], combo[2], combo[3], combo[4], combo[5], combo[6], combo[7]

        paths_list = [path_train]
        word_dict, pos_dict = dataset.get_vocabs(paths_list)
        train_dataset = dataset.PosDataset(word_dict, pos_dict, data_dir, 'train', padding=False)
        train_dataloader = DataLoader(train_dataset, shuffle=True)

        test_dataset = dataset.PosDataset(word_dict, pos_dict, data_dir, 'test', padding=False)
        test_dataloader = DataLoader(test_dataset, shuffle=False)

        word_vocab_size = len(train_dataset.word_idx_mappings)
        tag_vocab_size = len(train_dataset.pos_idx_mappings)

        base_model = models.BasicDependencyParserModel(word_vocab_size, tag_vocab_size, WORD_EMBEDDING_DIM,
                                                       TAG_EMBEDDING_DIM,
                                                       hidden_dim=LSTM_HIDDEN_DIM, mlp_dim_out=MLP_HIDDEN_DIM)

        res_dir = opj(net_results_dir, f"{combo}_adv_model_results" + time.strftime("%Y%m%d-%H%M%S"))

        if not os.path.isdir(res_dir):
            os.makedirs(res_dir)

        results_dict[combo] = train_net(base_model, train_dataloader, test_dataloader, nll_loss_func, EPOCHS=EPOCHS, BATCH_SIZE=BATCH_SIZE,
                  lr=LR, plot_progress=True, results_dir_path=res_dir, change_lr=CHANGE_LR, consider_sentence_len=True)

        with open(opj(net_results_dir, "final_combos_results.pkl"), 'wb') as f:
            pickle.dump(results_dict, f,  protocol=pickle.HIGHEST_PROTOCOL)

##################################################################################################################
def tag_comp_file(NET_PATH, tag_file_path, file_prefix="", model_type=0):
    """
    model_type: 0 - base model (word emb' trained only on words from train dataset), 1 - advanced model (word emb' trained on all the words)
    """
    WORD_EMBEDDING_DIM = 50
    if model_type == 0:
      paths_list = [path_train]
    elif model_type == 1:
      paths_list = [path_train, path_test, path_comp]
    else:
      raise Exception("Unknown model type")
      
    word_dict, pos_dict = dataset.get_vocabs(paths_list)
    comp_dataset = dataset.PosDataset(word_dict, pos_dict, data_dir, 'comp', padding=False,
                                      alpha_dropout=0.0, WORD_EMBD_DIM=WORD_EMBEDDING_DIM)
                                      
    comp_dataloader = DataLoader(comp_dataset, shuffle=False)

    model = models.AdvDependencyParserModel.load(NET_PATH)
    p = predict_from_loader(model, comp_dataloader)
    write_comp(tag_file_path, opj(data_dir, file_prefix + '.labeled'), p)

##################################################################################################################
if __name__ == '__main__':
    run_base_model()
    run_adv_model()
    #run_different_combos()
    #NET_PATH_base = r"/home/student/NLP_HW_2/submission/_epoch_6_acc_0.8852.pt"
    #NET_PATH_adv = r"/home/student/NLP_HW_2/submission/_epoch_6_acc_0.9042.pt"
    #tag_comp_file(NET_PATH_base, path_comp, "comp_m1_312088727", model_type=0)
    #tag_comp_file(NET_PATH_adv, path_comp, "comp_m2_312088727", model_type=1)




