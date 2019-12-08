
import argparse
import itertools
import os.path
import time
import uuid

import torch
import torch.optim.lr_scheduler

import numpy as np
import math
import json
from Datareader import syndep_reader
from Datareader import srlspan_reader
from Datareader import srldep_reader
import trees
import vocabulary
import makehp
import Zmodel
import utils

tokens = Zmodel

def count_wh(str, data):
    cun_w = 0
    for i, c_tree in enumerate(data):
        nodes = [c_tree]
        while nodes:
            node = nodes.pop()
            if isinstance(node, trees.InternalParseNode):
                cun_w += node.cun_w
                nodes.extend(reversed(node.children))

    print("total wrong head of :", str, "is", cun_w)

def align_sent(true_sents, wrong_sents, align_path):
    if not os.path.exists(align_path):
        align_dict = {}
        for i, t_sents in enumerate(true_sents):
            flag = 0
            for j, w_sents in enumerate(wrong_sents):
                #print(w_sents, t_sents)
                if w_sents == t_sents:
                    align_dict[i] = j
                    flag = 1
                    break
            if flag == 0:
                align_dict[i] = -1
            if j % 5000 == 0:
                print("done aligning", j)
        json.dump(align_dict, open(align_path, 'w'))
    else:
        with open(align_path, 'r') as f:
            align_dict = json.load(fp=f)

    return align_dict

def make_align(align_dict, sent_w, dict_w):
    sent = []
    dict = []
    for cun, i in align_dict.items():
        if i != -1:
            sent.append(sent_w[i])
            dict.append(dict_w[i])
        else:
            sent.append(None)
            dict.append(None)

    return  sent, dict

def correct_sent(syndep_sents, srlspan_sents, srldep_sents):
    for i, (syndep_sent, srlspan_sent, srldep_sent) in enumerate(zip(syndep_sents, srlspan_sents, srldep_sents)):

        assert len(syndep_sent) == len(srlspan_sent)
        if srldep_sent is not None:
            assert len(syndep_sent) == len(srldep_sent)

def span_miss_verb(srlspan_verb, srldep_verb):
    cun = 0
    for span_verb, dep_verb in zip(srlspan_verb, srldep_verb):
        dep_verb_list = [verb[0] for verb in dep_verb]
        for verb in span_verb:
            if verb not in dep_verb_list:
                cun += 1
    print("span miss verb ", cun)

class PTBDataset(object):
    def __init__(self, hparams):
        self.hparams = hparams
        self.tag_vocab = vocabulary.Vocabulary()
        self.word_vocab = vocabulary.Vocabulary()
        self.label_vocab = vocabulary.Vocabulary()
        self.type_vocab = vocabulary.Vocabulary()
        self.srl_vocab = vocabulary.Vocabulary()
        self.char_vocab = vocabulary.Vocabulary()
        self.ptb_dataset = {}
        self.max_line = 0
        self.dev_num = 0
        self.test_num = 0

    def rand_dataset(self):

        np.random.shuffle(self.ptb_dataset['train_data'])
        self.ptb_dataset['train_synconst_tree'] = [data[0] for data in self.ptb_dataset['train_data']]
        self.ptb_dataset['train_synconst_parse'] = [data[1] for data in self.ptb_dataset['train_data']]
        self.ptb_dataset['train_srlspan_dict'] = [data[2] for data in self.ptb_dataset['train_data']]
        self.ptb_dataset['train_srldep_dict'] = [data[3] for data in self.ptb_dataset['train_data']]

    def process_PTB(self, args):
        # srl dev set which uses 24 section of ptb is different from syn
        synconst_train_path = args.synconst_train_ptb_path
        synconst_dev_path = args.synconst_dev_ptb_path
        synconst_test_path = args.synconst_test_ptb_path

        syndep_train_path = args.syndep_train_ptb_path
        syndep_dev_path = args.syndep_dev_ptb_path
        syndep_test_path = args.syndep_test_ptb_path

        srlspan_train_path = args.srlspan_train_ptb_path
        srlspan_dev_path = args.srlspan_dev_ptb_path
        srlspan_test_path = args.srlspan_test_ptb_path
        srlspan_brown_path = args.srlspan_test_brown_path

        srldep_train_path = args.srldep_train_ptb_path
        srldep_dev_path = args.srldep_dev_ptb_path
        srldep_test_path = args.srldep_test_ptb_path
        srldep_brown_path = args.srldep_test_brown_path

        seldep_train_align_path = args.srldep_align_path #"data/seldep_train_align_path.json"

        self.ptb_dataset['train_syndep_sent'], self.ptb_dataset['train_syndep_head'], self.ptb_dataset['train_syndep_type'] = \
            syndep_reader.read_syndep( syndep_train_path, self.hparams.max_len_train)

        self.ptb_dataset['dev_syndep_sent'], self.ptb_dataset['dev_syndep_head'], self.ptb_dataset['dev_syndep_type'] = \
            syndep_reader.read_syndep( syndep_dev_path, self.hparams.max_len_dev)

        self.ptb_dataset['test_syndep_sent'], self.ptb_dataset['test_syndep_head'], self.ptb_dataset['test_syndep_type'] = \
            syndep_reader.read_syndep(syndep_test_path)

        self.ptb_dataset['train_srlspan_sent'], self.ptb_dataset['train_srlspan_dict'], self.ptb_dataset['train_srlspan_goldpos']\
            = srlspan_reader.read_srlspan(srlspan_train_path, self.hparams.max_len_train)

        self.ptb_dataset['dev_srlspan_sent'], self.ptb_dataset['dev_srlspan_dict'], self.ptb_dataset['dev_srlspan_goldpos']\
            = srlspan_reader.read_srlspan(srlspan_dev_path,self.hparams.max_len_dev)

        self.ptb_dataset['test_srlspan_sent'], self.ptb_dataset['test_srlspan_dict'], self.ptb_dataset['test_srlspan_goldpos']\
            = srlspan_reader.read_srlspan(srlspan_test_path)

        self.ptb_dataset['brown_srlspan_sent'], self.ptb_dataset['brown_srlspan_dict'],self.ptb_dataset['brown_srlspan_goldpos']\
            = srlspan_reader.read_srlspan(srlspan_brown_path)

        self.ptb_dataset['train_srldep_sent'], self.ptb_dataset['train_srldep_dict'], _ = srldep_reader.read_srldep(srldep_train_path, self.hparams.max_len_train)
        self.ptb_dataset['dev_srldep_sent'], self.ptb_dataset['dev_srldep_dict'], self.ptb_dataset['dev_srldep_pos'] = srldep_reader.read_srldep(srldep_dev_path, self.hparams.max_len_dev)
        self.ptb_dataset['test_srldep_sent'], self.ptb_dataset['test_srldep_dict'], self.ptb_dataset['test_srldep_pos'] = srldep_reader.read_srldep(srldep_test_path)
        self.ptb_dataset['brown_srldep_sent'], self.ptb_dataset['brown_srldep_dict'], self.ptb_dataset['brown_srldep_pos'] = srldep_reader.read_srldep(srldep_brown_path)

        print("aligning srl dep...")
        srldep_train_align_dict = align_sent(self.ptb_dataset['train_srlspan_sent'], self.ptb_dataset['train_srldep_sent'], seldep_train_align_path)
        self.ptb_dataset['train_srldep_sent'], self.ptb_dataset['train_srldep_dict'] = \
            make_align(srldep_train_align_dict, self.ptb_dataset['train_srldep_sent'], self.ptb_dataset['train_srldep_dict'])
        print("correct sents...")
        correct_sent(self.ptb_dataset['train_syndep_sent'], self.ptb_dataset['train_srlspan_sent'], self.ptb_dataset['train_srldep_sent'])

        print("Loading training trees from {}...".format(synconst_train_path))
        with open(synconst_train_path) as infile:
            treebank = infile.read()
        train_treebank = trees.load_trees(treebank, self.ptb_dataset['train_syndep_head'], self.ptb_dataset['train_syndep_type'],
                                          self.ptb_dataset['train_srlspan_goldpos'])
        if self.hparams.max_len_train > 0:
            train_treebank = [tree for tree in train_treebank if len(list(tree.leaves())) <= self.hparams.max_len_train]
        print("Loaded {:,} training examples.".format(len(train_treebank)))
        self.ptb_dataset['train_synconst_tree'] = train_treebank

        print("Loading development trees from {}...".format(synconst_dev_path))
        with open(synconst_dev_path) as infile:
            treebank = infile.read()
        dev_treebank = trees.load_trees(treebank, self.ptb_dataset['dev_syndep_head'], self.ptb_dataset['dev_syndep_type'])
        # different dev, srl is empty
        if self.hparams.max_len_dev > 0:
            dev_treebank = [tree for tree in dev_treebank if len(list(tree.leaves())) <= self.hparams.max_len_dev]
        print("Loaded {:,} development examples.".format(len(dev_treebank)))
        self.ptb_dataset['dev_synconst_tree'] = dev_treebank


        print("Loading test trees from {}...".format(synconst_test_path))
        with open(synconst_test_path) as infile:
            treebank = infile.read()
        test_treebank = trees.load_trees(treebank, self.ptb_dataset['test_syndep_head'], self.ptb_dataset['test_syndep_type'],
                                         self.ptb_dataset['test_srlspan_goldpos'])
        print("Loaded {:,} test examples.".format(len(test_treebank)))
        self.ptb_dataset['test_synconst_tree'] = test_treebank

        print("Processing trees for training...")
        self.ptb_dataset['train_synconst_parse'] = [tree.convert() for tree in train_treebank]
        dev_parse = [tree.convert() for tree in dev_treebank]
        test_parse = [tree.convert() for tree in test_treebank]


        count_wh("train data:", self.ptb_dataset['train_synconst_parse'])
        count_wh("dev data:", dev_parse)
        count_wh("test data:", test_parse)

        self.ptb_dataset['train_data'] = [(tree_bank, parse_tree, srlspan, srldep) for tree_bank, parse_tree, srlspan, srldep in
                      zip(self.ptb_dataset['train_synconst_tree'], self.ptb_dataset['train_synconst_parse'], self.ptb_dataset['train_srlspan_dict'], self.ptb_dataset['train_srldep_dict'])]

        print("Constructing vocabularies...")

        self.tag_vocab.index(Zmodel.START)
        self.tag_vocab.index(Zmodel.STOP)
        self.tag_vocab.index(Zmodel.TAG_UNK)

        self.word_vocab.index(Zmodel.START)
        self.word_vocab.index(Zmodel.STOP)
        self.word_vocab.index(Zmodel.UNK)

        self.label_vocab.index(())
        sublabels = [Zmodel.Sub_Head]
        self.label_vocab.index(tuple(sublabels))

        self.type_vocab = vocabulary.Vocabulary()

        self.srl_vocab.index('*')

        for srl_dict in self.ptb_dataset['train_srldep_dict']:
            if srl_dict is not None:
                for verb_id, arg_list in srl_dict.items():
                    for arg in arg_list:
                        self.srl_vocab.index(arg[1])

        for srl_dict in self.ptb_dataset['train_srlspan_dict']:
            if srl_dict is not None:
                for verb_id, arg_list in srl_dict.items():
                    for arg in arg_list:
                        self.srl_vocab.index(arg[2])

        char_set = set()

        for i, tree in enumerate(self.ptb_dataset['train_synconst_parse']):

            const_sentences = [leaf.word for leaf in tree.leaves()]
            assert len(const_sentences) == len(self.ptb_dataset['train_syndep_sent'][i])
            assert len(const_sentences) == len(self.ptb_dataset['train_srlspan_sent'][i])
            nodes = [tree]
            while nodes:
                node = nodes.pop()
                if isinstance(node, trees.InternalParseNode):
                    self.label_vocab.index(node.label)
                    nodes.extend(reversed(node.children))
                else:
                    self.tag_vocab.index(node.tag)
                    self.word_vocab.index(node.word)
                    self.type_vocab.index(node.type)
                    char_set |= set(node.word)

        # char_vocab.index(tokens.CHAR_PAD)

        # If codepoints are small (e.g. Latin alphabet), index by codepoint directly
        highest_codepoint = max(ord(char) for char in char_set)
        if highest_codepoint < 512:
            if highest_codepoint < 256:
                highest_codepoint = 256
            else:
                highest_codepoint = 512

            # This also takes care of constants like tokens.CHAR_PAD
            for codepoint in range(highest_codepoint):
                char_index = self.char_vocab.index(chr(codepoint))
                assert char_index == codepoint
        else:
            self.char_vocab.index(tokens.CHAR_UNK)
            self.char_vocab.index(tokens.CHAR_START_SENTENCE)
            self.char_vocab.index(tokens.CHAR_START_WORD)
            self.char_vocab.index(tokens.CHAR_STOP_WORD)
            self.char_vocab.index(tokens.CHAR_STOP_SENTENCE)
            for char in sorted(char_set):
                self.char_vocab.index(char)

        self.tag_vocab.freeze()
        self.word_vocab.freeze()
        self.label_vocab.freeze()
        self.char_vocab.freeze()
        self.type_vocab.freeze()
        self.srl_vocab.freeze()

        def print_vocabulary(name, vocab):
            special = {tokens.START, tokens.STOP, tokens.UNK}
            print("{} ({:,}): {}".format(
                name, vocab.size,
                sorted(value for value in vocab.values if value in special) +
                sorted(value for value in vocab.values if value not in special)))

        if args.print_vocabs:
            print_vocabulary("Tag", self.tag_vocab)
            print_vocabulary("Word", self.word_vocab)
            print_vocabulary("Label", self.label_vocab)
            print_vocabulary("Char", self.char_vocab)
            print_vocabulary("Type", self.type_vocab)
            print_vocabulary("Srl", self.srl_vocab)

        self.max_line = len(self.ptb_dataset['train_data'])
        self.ptb_dataset['dev_synconst'] = [[(leaf.tag, leaf.word) for leaf in tree.leaves()] for tree in dev_treebank]
        self.ptb_dataset['test_synconst'] = [[(leaf.tag, leaf.word) for leaf in tree.leaves()] for tree in test_treebank]

        self.ptb_dataset['dev_srlspan'] = [[(tag, word) for j, (tag, word) in enumerate(zip(tags, words))]
                                           for i,(tags, words) in enumerate(zip(self.ptb_dataset['dev_srlspan_goldpos'], self.ptb_dataset['dev_srlspan_sent']))]

        self.ptb_dataset['test_srlspan'] = [[(tag, word) for j, (tag, word) in enumerate(zip(tags, words))]
                                           for i,(tags, words) in enumerate(zip(self.ptb_dataset['test_srlspan_goldpos'], self.ptb_dataset['test_srlspan_sent']))]

        self.ptb_dataset['brown_srlspan'] = [[(tag, word) for j, (tag, word) in enumerate(zip(tags, words))]
                                            for i, (tags, words) in enumerate(
                zip(self.ptb_dataset['brown_srlspan_goldpos'], self.ptb_dataset['brown_srlspan_sent']))]


        self.ptb_dataset['dev_srldep'] = [[(tag, word) for j, (tag, word) in enumerate(zip(tags, words))]
                                           for i, (tags, words) in enumerate(
                zip(self.ptb_dataset['dev_srldep_pos'], self.ptb_dataset['dev_srldep_sent']))]

        self.ptb_dataset['test_srldep'] = [[(tag, word) for j, (tag, word) in enumerate(zip(tags, words))]
                                           for i, (tags, words) in enumerate(
                zip(self.ptb_dataset['test_srldep_pos'], self.ptb_dataset['test_srldep_sent']))]

        self.ptb_dataset['brown_srldep'] = [[(tag, word) for j, (tag, word) in enumerate(zip(tags, words))]
                                           for i, (tags, words) in enumerate(
                zip(self.ptb_dataset['brown_srldep_pos'], self.ptb_dataset['brown_srldep_sent']))]

        length = 0
        for i in range(10):
            length += 10

            self.ptb_dataset['dev_synconst_tree'+ str(length)] = [tree for tree in dev_treebank if len(list(tree.leaves())) <= length and len(list(tree.leaves())) > length - 10]

            self.ptb_dataset['dev_synconst'+ str(length)] = [[(leaf.tag, leaf.word) for leaf in tree.leaves()] for tree in
                                                             self.ptb_dataset['dev_synconst_tree' + str(length)]]

            self.ptb_dataset['dev_syndep_sent'+ str(length)] = [data for data in self.ptb_dataset['dev_syndep_sent'] if len(data) <= length and len(data) > length - 10]
            self.ptb_dataset['dev_syndep_head' + str(length)] = [data for data in self.ptb_dataset['dev_syndep_head'] if
                                                                 len(data) <= length and len(data) > length - 10]
            self.ptb_dataset['dev_syndep_type' + str(length)] = [data for data in self.ptb_dataset['dev_syndep_type'] if
                                                                 len(data) <= length and len(data) > length - 10]


            self.ptb_dataset['dev_srlspan_goldpos' + str(length)] = [data for data in self.ptb_dataset['dev_srlspan_goldpos'] if len(data) <= length and len(data) > length - 10]

            self.ptb_dataset['dev_srlspan_sent' + str(length)]= []
            self.ptb_dataset['dev_srlspan_dict' + str(length)] = []
            for (sent, dict) in zip(self.ptb_dataset['dev_srlspan_sent'],self.ptb_dataset['dev_srlspan_dict']):
                if len(sent) <= length and len(sent) > length - 10:
                    self.ptb_dataset['dev_srlspan_sent' + str(length)].append(sent)
                    self.ptb_dataset['dev_srlspan_dict' + str(length)].append(dict)

            self.ptb_dataset['dev_srlspan'+ str(length)] = [[(tag, word) for j, (tag, word) in enumerate(zip(tags, words))]
                                               for i, (tags, words) in enumerate(
                    zip(self.ptb_dataset['dev_srlspan_goldpos'+ str(length)], self.ptb_dataset['dev_srlspan_sent'+ str(length)]))]

            self.ptb_dataset['dev_srldep_sent' + str(length)] = []
            self.ptb_dataset['dev_srldep_dict' + str(length)] = []
            for (sent, dict) in zip(self.ptb_dataset['dev_srldep_sent'], self.ptb_dataset['dev_srldep_dict']):
                if len(sent) <= length and len(sent) > length - 10:
                    self.ptb_dataset['dev_srldep_sent' + str(length)].append(sent)
                    self.ptb_dataset['dev_srldep_dict' + str(length)].append(dict)

            self.ptb_dataset['dev_srldep_pos' + str(length)] = [data for data in
                                                                     self.ptb_dataset['dev_srldep_pos'] if
                                                                     len(data) <= length and len(data) > length - 10]

            self.ptb_dataset['dev_srldep'+ str(length)] = [[(tag, word) for j, (tag, word) in enumerate(zip(tags, words))]
                                              for i, (tags, words) in enumerate(
                    zip(self.ptb_dataset['dev_srldep_pos'+ str(length)], self.ptb_dataset['dev_srldep_sent'+ str(length)]))]


