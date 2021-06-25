import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
from torch.utils.data.dataloader import DataLoader
from collections import defaultdict
from torchtext.vocab import Vocab
from torch.utils.data.dataset import Dataset, TensorDataset
from pathlib import Path
from collections import Counter
from os.path import join as opj


data_dir = r"C:\git-projects\NLP_HW_2\data"
path_train = opj(data_dir, "train.labeled")
path_test = opj(data_dir, "test.labeled")

UNKNOWN_TOKEN = "<unk>"
PAD_TOKEN = "<pad>"  # Optional: this is used to pad a batch of sentences in different lengths.
# ROOT_TOKEN = PAD_TOKEN # this can be used if you are not padding your batches
ROOT_TOKEN = "<root>" # use this if you are padding your batches and want a special token for ROOT
SPECIAL_TOKENS = [ROOT_TOKEN, PAD_TOKEN, UNKNOWN_TOKEN]

########################################################################################
def split(string, delimiters):
    """
        Split strings according to delimiters
        :param string: full sentence
        :param delimiters string: characters for spliting
            function splits sentence to words
    """
    delimiters = tuple(delimiters)
    stack = [string, ]

    for delimiter in delimiters:
        for i, substring in enumerate(stack):
            substack = substring.split(delimiter)
            stack.pop(i)
            for j, _substring in enumerate(substack):
                stack.insert(i + j, _substring)

    return stack

########################################################################################
def get_vocabs(list_of_paths):
    """
        Extract vocabs from given datasets. Return a word2ids and tag2idx.
        :param file_paths: a list with a full path for all corpuses
            Return:
              - word2idx
              - tag2idx
    """
    word_dict = defaultdict(int)
    pos_dict = defaultdict(int)
    for file_path in list_of_paths:
        with open(file_path) as f:
            for line in f:
                if line == "\n":
                    continue

                splited_words = split(line, (' ', '\n', '\t', '_'))

                while '' in splited_words:
                    splited_words.remove('')

                del splited_words[-1]
                word = splited_words[1].lower()
                pos_tag = splited_words[2]

                word_dict[word] += 1
                pos_dict[pos_tag] += 1

    return word_dict, pos_dict

########################################################################################
class PosDataReader:
    def __init__(self, file, word_dict, pos_dict):
        self.file = file
        self.word_dict = word_dict
        self.pos_dict = pos_dict
        self.sentences = []
        self.__readData__()

    def __readData__(self):
        """main reader function which also populates the class data structures"""
        cur_sentence = []
        with open(self.file, 'r') as f:
            for line in f:
                if line == "\n":
                    self.sentences.append(cur_sentence)
                    cur_sentence = [(ROOT_TOKEN, ROOT_TOKEN, -1)]
                    #cur_sentence = []
                    continue

                splited_words = split(line, (' ', '\n', '\t', '_'))

                while '' in splited_words:
                    splited_words.remove('')
                del splited_words[-1]

                word = splited_words[1].lower()
                pos_tag = splited_words[2]
                head_idx = int(splited_words[3])

                cur_sentence.append((word, pos_tag, head_idx))

    def get_num_sentences(self):
        """returns num of sentences in data"""
        return len(self.sentences)

########################################################################################


class PosDataset(Dataset):
    def __init__(self, word_dict, pos_dict, dir_path: str, subset: str,
                 padding=False, word_embeddings=None, alpha_dropout=0.25, WORD_EMBD_DIM=300):
        """
        :param word_dict:
        :param pos_dict:
        :param dir_path:
        :param subset:
        :param padding:
        :param word_embeddings:
        """
        super().__init__()

        self.alpha_dropout = alpha_dropout

        assert subset in ['train', 'test']
        self.subset = subset

        self.file = opj(dir_path, subset + ".labeled")

        self.datareader = PosDataReader(self.file, word_dict, pos_dict)

        self.vocab_size = len(self.datareader.word_dict)

        if word_embeddings:
            self.word_idx_mappings, self.idx_word_mappings, self.word_vectors = word_embeddings
        else:
            self.word_idx_mappings, self.idx_word_mappings, self.word_vectors =\
                self.init_word_embeddings(self.datareader.word_dict, WORD_EMBD_DIM)

        self.pos_idx_mappings, self.idx_pos_mappings = self.init_pos_vocab(self.datareader.pos_dict)

        self.pad_idx = self.word_idx_mappings.get(PAD_TOKEN)
        self.unknown_idx = self.word_idx_mappings.get(UNKNOWN_TOKEN)
        self.word_vector_dim = self.word_vectors.size(-1)
        self.sentence_lens = [len(sentence) for sentence in self.datareader.sentences]
        self.max_seq_len = max(self.sentence_lens)
        self.sentences_dataset = self.convert_sentences_to_dataset(padding)

    def __len__(self):
        return len(self.sentences_dataset)

    def __getitem__(self, index):
        word_embed_idx, pos_embed_idx, head_idx, sentence_len = self.sentences_dataset[index]
        return word_embed_idx, pos_embed_idx, head_idx, sentence_len

    @staticmethod
    def init_word_embeddings(word_dict, WORD_EMBD_DIM):
        glove = Vocab(Counter(word_dict), vectors=f"glove.6B.{WORD_EMBD_DIM}d", specials=SPECIAL_TOKENS)
        return glove.stoi, glove.itos, glove.vectors

    def get_word_embeddings(self):
        return self.word_idx_mappings, self.idx_word_mappings, self.word_vectors

    def init_pos_vocab(self, pos_dict):
        idx_pos_mappings = sorted([self.word_idx_mappings.get(token) for token in SPECIAL_TOKENS])
        pos_idx_mappings = {self.idx_word_mappings[idx]: idx for idx in idx_pos_mappings}

        for i, pos in enumerate(sorted(pos_dict.keys())):
            # pos_idx_mappings[str(pos)] = int(i)
            pos_idx_mappings[str(pos)] = int(i + len(SPECIAL_TOKENS))
            idx_pos_mappings.append(str(pos))
        #print("idx_pos_mappings -", idx_pos_mappings)
        #print("pos_idx_mappings -", pos_idx_mappings)
        return pos_idx_mappings, idx_pos_mappings

    def get_pos_vocab(self):
        return self.pos_idx_mappings, self.idx_pos_mappings

    def convert_sentences_to_dataset(self, padding):
        sentence_word_idx_list = list()
        sentence_pos_idx_list = list()
        sentence_head_idx_list = list()
        sentence_len_list = list()
        for sentence_idx, sentence in enumerate(self.datareader.sentences):
            words_idx_list = []
            pos_idx_list = []
            head_idx_list = []
            for word, pos, head_idx in sentence:
                word_app = self.datareader.word_dict.get(word)
                word_app = 0 if word_app is None else word_app
                if self.alpha_dropout > 0 and word != ROOT_TOKEN:
                    prob_for_drop = self.alpha_dropout / (word_app + self.alpha_dropout)
                    if prob_for_drop > np.random.rand():
                        word = UNKNOWN_TOKEN

                if self.pos_idx_mappings.get(pos) is None and pos != ROOT_TOKEN:
                    pos = UNKNOWN_TOKEN

                words_idx_list.append(self.word_idx_mappings.get(word))
                pos_idx_list.append(self.pos_idx_mappings.get(pos))
                head_idx_list.append(head_idx)

            sentence_len = len(words_idx_list)
            # if padding:
            #     while len(words_idx_list) < self.max_seq_len:
            #         words_idx_list.append(self.word_idx_mappings.get(PAD_TOKEN))
            #         pos_idx_list.append(self.pos_idx_mappings.get(PAD_TOKEN))

            sentence_word_idx_list.append(torch.tensor(words_idx_list, dtype=torch.long, requires_grad=False))
            sentence_pos_idx_list.append(torch.tensor(pos_idx_list, dtype=torch.long, requires_grad=False))
            sentence_head_idx_list.append(torch.tensor(head_idx_list, dtype=torch.long, requires_grad=False))
            sentence_len_list.append(sentence_len)

        # if padding:
        #     all_sentence_word_idx = torch.tensor(sentence_word_idx_list, dtype=torch.long)
        #     all_sentence_pos_idx = torch.tensor(sentence_pos_idx_list, dtype=torch.long)
        #     all_sentence_len = torch.tensor(sentence_len_list, dtype=torch.long, requires_grad=False)
        #     return TensorDataset(all_sentence_word_idx, all_sentence_pos_idx, all_sentence_len)

        return {i: sample_tuple for i, sample_tuple in enumerate(zip(sentence_word_idx_list,
                                                                     sentence_pos_idx_list,
                                                                     sentence_head_idx_list,
                                                                     sentence_len_list))}

########################################################################################



