
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import logging
import argparse
from tqdm import tqdm, trange
import json

import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler
from torch.utils.data.distributed import DistributedSampler

from torch.utils.data import Dataset
import random
import trees
import copy

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


def warmup_linear(x, warmup=0.002):
    if x < warmup:
        return x/warmup
    return 1.0 - x

BERT_TOKEN_MAPPING = {
    "-LRB-": "(",
    "-RRB-": ")",
    "-LCB-": "{",
    "-RCB-": "}",
    "-LSB-": "[",
    "-RSB-": "]",
    "``": '"',
    "''": '"',
    "`": "'",
    '«': '"',
    '»': '"',
    '‘': "'",
    '’': "'",
    '“': '"',
    '”': '"',
    '„': '"',
    '‹': "'",
    '›': "'",
    }
class BERTDataset(Dataset):
    def __init__(self, pre_wiki_line, hparams, ptb_dataset, corpus_path, tokenizer, seq_len, encoding="utf-8", corpus_lines=None, on_memory=True):
        if hparams.model =="xlnet":
            self.vocab = tokenizer.sp_model
        else:
            self.vocab = tokenizer.vocab
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.on_memory = on_memory
        self.corpus_lines = corpus_lines  # number of non-empty lines in input corpus
        self.total_lines = 0
        self.wiki_id = 0 #number of wiki data
        self.wiki_line = 0
        self.pre_wiki_line = pre_wiki_line
        if self.wiki_line < self.pre_wiki_line:
            self.init_wiki = True
        else:
            self.init_wiki = False
        self.corpus_path = corpus_path #one line is a dict
        # self.jointdata_path = jointdata_path
        self.encoding = encoding
        self.current_doc = 0  # to avoid random sentence from same doc

        self.ptb_dataset = ptb_dataset
        self.ptb_cur_line = 0
        self.ptb_max_line = ptb_dataset.max_line
        self.ptb_epoch = 0

        self.hparams = hparams
        if hparams.bert_transliterate:
            from transliterate import TRANSLITERATIONS
            self.bert_transliterate = TRANSLITERATIONS[hparams.bert_transliterate]
        else:
            self.bert_transliterate = None

        # for loading samples directly from file
        self.sample_counter = 0  # used to keep track of full epochs on file
        self.line_buffer = None  # keep second sentence of a pair in memory and use as first sentence in next pair
        self.line_pre = None #  albert previous line

        # for loading samples in memory
        self.current_random_doc = 0
        self.num_docs = 0
        self.sample_to_doc = [] # map sample index to doc and line

        # load samples into memory
        if on_memory:
            self.all_docs = []
            doc = []
            self.corpus_lines = 0
            with open(corpus_path, "r", encoding=encoding) as f:
                for line in tqdm(f, desc="Loading Dataset", total=corpus_lines):
                    line_as = line.strip()
                    if line_as == "":
                        self.all_docs.append(doc)
                        doc = []
                        #remove last added sample because there won't be a subsequent line anymore in the doc
                        self.sample_to_doc.pop()
                    else:
                        #store as one sample
                        sample = {"doc_id": len(self.all_docs),
                                  "line": len(doc)}
                        self.sample_to_doc.append(sample)
                        doc.append(line)
                        self.corpus_lines = self.corpus_lines + 1

            # if last row in file is not empty
            if self.all_docs[-1] != doc:
                self.all_docs.append(doc)
                self.sample_to_doc.pop()

            self.num_docs = len(self.all_docs)

        # load samples later lazily from disk
        else:
            if self.corpus_lines is None:
                with open(corpus_path, "r", encoding=encoding) as f:
                    self.corpus_lines = 0
                    for line in tqdm(f, desc="Loading Dataset", total=corpus_lines):
                        self.total_lines += 1
                        if line.strip() == "":
                            self.num_docs += 1
                        else:
                            self.corpus_lines += 1

                    # if doc does not end with empty line
                    if line.strip() != "":
                        self.num_docs += 1
            print("total_lines", self.total_lines)
            self.file = open(corpus_path, "r", encoding=encoding)
            self.random_file = open(corpus_path, "r", encoding=encoding)
            if self.init_wiki:
                while self.wiki_line < self.pre_wiki_line:
                    self.file.__next__().strip()
                    self.wiki_line += 1
                self.init_wiki = False
            # self.joint_file = open(jointdata_path, "r", encoding=encoding)

    def __len__(self):
        # last line of doc won't be used, because there's no "nextSentence". Additionally, we start counting at 0.
        return self.corpus_lines - self.num_docs - 1

    def __getitem__(self, item):
        cur_id = self.sample_counter
        self.sample_counter += 1

        # if not self.on_memory:
        #     # after one epoch we start again from beginning of file
        #     if self.wiki_id != 0 and (self.wiki_id % len(self) == 0):
        #         self.wiki_id = 0
        #         self.file.close()
        #         self.file = open(self.corpus_path, "r", encoding=self.encoding)

        # t1, t2, is_next_label = self.random_sent(item)

        too_long = True
        cur_item = item
        is_single = False
        while too_long or is_single:

            t1, t2, is_next_label, dict1, is_ptb = self.get_joint_data(cur_item)
            tokens_a, word_list_a, const_list_a, srl_list_a, word_start_mask, word_end_mask, tokenlength_a = self.token_span(t1)
            if isinstance(t1[2], trees.LeafParseNode):
                is_single = True
            else:
                is_single = False
            cur_item += 1
            if cur_item >= self.corpus_lines:
                cur_item = 0

            # if not is_ptb:
            #     tokens_b, word_list_b, const_list_b, srl_list_b, _, _, tokenlength_b = self.token_span(t2)
            # else:
            #     tokenlength_b = 0
            if self.seq_len < tokenlength_a + 3:
                too_long = True
            else:
                too_long = False

        #just use first sent for syntax, srl tasks
        if not is_ptb:
            tokens_b, word_list_b, const_list_b, srl_list_b, _, _, _ = self.token_span(t2)
            cur_example = InputExample(guid=cur_id, tokens_a=tokens_a, word_a = word_list_a, const_a = const_list_a, srl_a = srl_list_a,
                                       tokens_b=tokens_b, word_b = word_list_b, const_b = const_list_b, srl_b = srl_list_b, is_next=is_next_label)
        else:
            cur_example = InputExample(guid=cur_id, tokens_a=tokens_a, word_a=word_list_a, const_a=const_list_a, srl_a=srl_list_a, is_next=is_next_label)
            # combine to one sample

        # transform sample to features
        cur_features = self.convert_example_to_features(cur_example, word_start_mask, word_end_mask, self.seq_len, self.tokenizer, is_ptb)

        # cur_data = (t1, t2, is_next_label)

        # print(dict1)
        cur_data = (torch.tensor(cur_features.input_ids),#.requires_grad_(False),
                    torch.tensor(cur_features.origin_ids),#.requires_grad_(False),
                    torch.tensor(cur_features.input_mask),
                    torch.tensor(cur_features.word_start_mask),
                    torch.tensor(cur_features.word_end_mask),
                    torch.tensor(cur_features.segment_ids),
                    torch.tensor(cur_features.perm_mask),
                    torch.tensor(cur_features.target_mapping),
                    torch.tensor(cur_features.lm_label_ids),
                    torch.tensor(cur_features.lm_label_mask),
                    torch.tensor(cur_features.is_next),
                    # dict1,
                    dict1['synconst'],
                    dict1['syndep_head'],
                    dict1['syndep_type'],
                    dict1['srlspan'],
                    dict1['srldep'],
                    is_ptb,
                    )

        return cur_data #, (dict1['synconst'],dict1['syndep_head'],dict1['syndep_type'],dict1['srlspan'],dict1['srldep'])

    def token_span(self, joint_data):

        sent, tree, parse, srlspan, srldep = joint_data
        too_long = False
        tokens = []
        word_start_mask = []
        word_end_mask = []
        word_start_idx = []     #for token idx
        word_end_idx = []
        idx = 0
        if not self.hparams.model =="xlnet":
            # tokens.append("[CLS]")
            word_start_mask.append(1)
            word_end_mask.append(1)
        if self.bert_transliterate is None:
            cleaned_words = []
            for _, word in sent:
                word = BERT_TOKEN_MAPPING.get(word, word)
                if word == "n't" and cleaned_words:
                    cleaned_words[-1] = cleaned_words[-1] + "n"
                    word = "'t"
                cleaned_words.append(word)
        else:
            # When transliterating, assume that the token mapping is
            # taken care of elsewhere
            cleaned_words = [self.bert_transliterate(word) for _, word in sent]

        for word in cleaned_words:
            word_tokens = self.tokenizer.tokenize(word)
            if len(word_tokens) == 0:
                word_tokens = [self.tokenizer.unk_token]

            # if self.seq_len < len(word_tokens) + len(tokens) + 1:
            #     too_long = True
            #     break

            word_start_idx.append(idx)
            for _ in range(len(word_tokens)):
                word_start_mask.append(0)
                word_end_mask.append(0)
                idx += 1
            word_end_idx.append(idx - 1)
            if self.hparams.model =="xlnet":
                word_start_mask[len(tokens)] = 1
            else:
                word_start_mask[len(tokens)+ 1] = 1 #since cls in the first token
            word_end_mask[-1] = 1
            tokens.extend(word_tokens)
        # tokens.append("[SEP]")
        word_start_mask.append(1)
        word_end_mask.append(1)

        if self.hparams.model =="xlnet":
            # tokens.append("[CLS]")
            word_start_mask.append(1)
            word_end_mask.append(1)

        word_list = []
        for st, en in zip(word_start_idx, word_end_idx):
            word_list.append((st, en))


        const_list = []
        nodes = [tree]
        while nodes:
            node = nodes.pop()
            if isinstance(node, trees.InternalTreebankNode):
                if node.right - 1 < len(word_end_idx):
                    const_list.append((word_start_idx[node.left], word_end_idx[node.right - 1]))
                nodes.extend(reversed(node.children))

        srl_list = []
        for pred_id, args in srlspan.items():
            srl_list += [(word_start_idx[a[0]], word_end_idx[a[1]]) for a in args if a[1] < len(word_end_idx)]

        token_length = len(tokens)

        return tokens, word_list, const_list, srl_list, word_start_mask, word_end_mask, token_length

    def get_joint_data(self, index):

        is_ptb = False

        if random.random() < self.hparams.p_ptb:

            #use ptb dataset

            if self.ptb_cur_line >= self.ptb_max_line:
                self.ptb_dataset.rand_dataset()
                self.ptb_cur_line = 0
                self.ptb_epoch += 1

            t1_syntree = self.ptb_dataset.ptb_dataset['train_synconst_tree'][self.ptb_cur_line]
            t1_srlspan = self.ptb_dataset.ptb_dataset['train_srlspan_dict'][self.ptb_cur_line]
            t1_srldep = self.ptb_dataset.ptb_dataset['train_srldep_dict'][self.ptb_cur_line]
            t1_sent = [(leaf.tag, leaf.word) for leaf in t1_syntree.leaves()]
            t1_synparse = self.ptb_dataset.ptb_dataset['train_synconst_parse'][self.ptb_cur_line]
            dict1 = {}
            dict1['synconst'] = t1_syntree.linearize()
            dict1['syndep_head'] = json.dumps([leaf.father for leaf in t1_syntree.leaves()])
            dict1['syndep_type'] = json.dumps([leaf.type for leaf in t1_syntree.leaves()])

            srlspan_str = {}
            for pred_id, args in t1_srlspan.items():
                srlspan_str[str(pred_id)] = [(str(a[0]), str(a[1]), a[2]) for a in args]

            srldep_str = {}
            if t1_srldep is None:
                srldep_str[str(-1)] = [(str(-1), str(-1))]
            else:
                for pred_id, args in t1_srldep.items():
                    srldep_str[str(pred_id)] = [(str(a[0]), a[1]) for a in args]

            dict1['srlspan'] = json.dumps(srlspan_str)
            dict1['srldep'] = json.dumps(srldep_str)
            # dict1 = json.dumps(dict_ptb)
            self.ptb_cur_line += 1

            # next_line = self.ptb_cur_line
            # if next_line >= self.ptb_max_line:
            #     # self.ptb_dataset.rand_dataset()
            #     next_line = 0
            #
            # t2_syntree = self.ptb_dataset.ptb_dataset['train_synconst_tree'][next_line]
            # t2_srlspan = self.ptb_dataset.ptb_dataset['train_srlspan_dict'][next_line]
            # t2_srldep = self.ptb_dataset.ptb_dataset['train_srldep_dict'][next_line]
            # t2_sent = [(leaf.tag, leaf.word) for leaf in t2_syntree.leaves()]
            # t2_synparse = self.ptb_dataset.ptb_dataset['train_synconst_parse'][next_line]
            # self.ptb_cur_line += 1

            t1 = (t1_sent, t1_syntree, t1_synparse, t1_srlspan, t1_srldep)
            # t2 = (t2_sent, t2_syntree, t2_synparse, t2_srlspan, t2_srldep)
            t2 = None
            label = -1
            is_ptb = True

        else:

            # use wiki dataset

            if not self.on_memory:
                # after one epoch we start again from beginning of file
                if self.wiki_line >= self.total_lines - 2:
                    self.wiki_line = 0
                    self.file.close()
                    self.file = open(self.corpus_path, "r", encoding=self.encoding)

            t0, t1, t2, dict1 = self.get_corpus_line(index)
            if random.random() > 0.5:
                label = 0
            else:
                if self.hparams.model =="albert":
                    t2 = t0
                else:
                    t2 = self.get_random_line()
                label = 1

            assert len(t1[0]) > 0
            assert len(t2[0]) > 0

            self.wiki_id += 1

        return t1, t2, label, dict1, is_ptb

    def dict_to_data(self, dict):
        synconst = dict['synconst']
        syndep_head = dict['syndep_head']
        syndep_type = dict['syndep_type']
        srlspan_str = dict['srlspan']
        srlspan = {}
        for pred_id, args in srlspan_str.items():
            srlspan[int(pred_id)] = [(int(a[0]), int(a[1]), a[2]) for a in args]

        srldep_str = dict['srldep']
        srldep = {}
        for pred_id, args in srldep_str.items():
            srldep[int(pred_id)] = [(int(a[0]), a[1]) for a in args]

        syntree = trees.load_trees(synconst, [[int(head) for head in syndep_head]], [syndep_type], strip_top = False)[0]
        sent = [(leaf.tag, leaf.word) for leaf in syntree.leaves()]
        synparse = syntree.convert()

        dict_new = {}
        dict_new['synconst'] = synconst
        dict_new['syndep_head'] = json.dumps(syndep_head)
        dict_new['syndep_type'] = json.dumps(syndep_type)
        dict_new['srlspan'] = json.dumps(srlspan_str)
        dict_new['srldep'] = json.dumps(srldep_str)

        return (sent, syntree, synparse, srlspan, srldep), dict_new


    def get_corpus_line(self, item):
        """
        Get one sample from corpus consisting of a pair of two subsequent lines from the same doc.
        :param item: int, index of sample.
        :return: (str, str), two subsequent sentences from corpus
        """
        t1 = ""
        t2 = ""
        assert item < self.corpus_lines
        if self.on_memory:
            sample = self.sample_to_doc[item]
            dict1 = self.all_docs[sample["doc_id"]][sample["line"]]
            dict2 = self.all_docs[sample["doc_id"]][sample["line"] + 1]
            # used later to avoid random nextSentence from same doc
            self.current_doc = sample["doc_id"]
            dict1 = json.loads(dict1)
            dict2 = json.loads(dict2)
            t1, dict_new1 = self.dict_to_data(dict1)
            t2, _ = self.dict_to_data(dict2)
            return t1, t2, dict1
        else:
            if self.line_buffer is None or self.line_pre is None:
                # read first non-empty line of file
                while t1 == "":
                    t0 = self.file.__next__().strip()
                    t1 = self.file.__next__().strip()
                    t2 = self.file.__next__().strip()
                    self.wiki_line += 1
                    if self.wiki_line >= self.total_lines - 2:
                        self.wiki_line = 0
                        self.file.close()
                        self.file = open(self.corpus_path, "r", encoding=self.encoding)
            else:
                # use t2 from previous iteration as new t1
                t0 = self.line_pre
                t1 = self.line_buffer
                t2 = self.file.__next__().strip()
                self.wiki_line += 1
                if self.wiki_line >= self.total_lines - 2:
                    self.wiki_line = 0
                    self.file.close()
                    self.file = open(self.corpus_path, "r", encoding=self.encoding)
            # skip empty rows that are used for separating documents and keep track of current doc id
            while t2 == "" or t1 == "" or t0 == "":
                t0 = self.file.__next__().strip()
                t1 = self.file.__next__().strip()
                t2 = self.file.__next__().strip()
                self.wiki_line += 1
                if self.wiki_line >= self.total_lines- 2:
                    self.wiki_line = 0
                    self.file.close()
                    self.file = open(self.corpus_path, "r", encoding=self.encoding)
                self.current_doc = self.current_doc + 1

            self.line_buffer = t2
            self.line_pre = t1

            # if self.init_wiki:
            #     return t1, t2, None
            dict0 = eval(t0)  # json.loads(t1)
            dict1 = eval(t1) #json.loads(t1)
            dict2 = eval(t2) #json.loads(t2)
            t0, dict_new0 = self.dict_to_data(dict0)
            t1, dict_new1 = self.dict_to_data(dict1)
            t2, _ = self.dict_to_data(dict2)

        assert t1[0] != ""
        assert t2[0] != ""
        return t0, t1, t2, dict_new1


    def get_random_line(self):
        """
        Get random line from another document for nextSentence task.
        :return: str, content of one line
        """
        # Similar to original tf repo: This outer loop should rarely go for more than one iteration for large
        # corpora. However, just to be careful, we try to make sure that
        # the random document is not the same as the document we're processing.
        line = ""
        while line == "":
            for _ in range(10):
                if self.on_memory:
                    rand_doc_idx = random.randint(0, len(self.all_docs)-1)
                    rand_doc = self.all_docs[rand_doc_idx]
                    line = rand_doc[random.randrange(len(rand_doc))]
                else:
                    rand_index = random.randint(1, self.corpus_lines if self.corpus_lines < 1000 else 1000)
                    #pick random line
                    for _ in range(rand_index):
                        line = self.get_next_line()
                #check if our picked random line is really from another doc like we want it to be
                if self.current_random_doc != self.current_doc:
                    break

        dict = json.loads(line)
        t1, _ = self.dict_to_data(dict)

        return t1

    def get_next_line(self):
        """ Gets next line of random_file and starts over when reaching end of file"""
        try:
            line = self.random_file.__next__().strip()
            #keep track of which document we are currently looking at to later avoid having the same doc as t1
            if line == "":
                self.current_random_doc = self.current_random_doc + 1
                line = self.random_file.__next__().strip()
        except StopIteration:
            self.random_file.close()
            self.random_file = open(self.corpus_path, "r", encoding=self.encoding)
            line = self.random_file.__next__().strip()
        return line

    def random_mask(self, tokens, word_list, const_list, srl_list, tokenizer):

        prob = random.random()
        p_sum = self.hparams.p_constmask + self.hparams.p_srlmask + self.hparams.p_wordmask + self.hparams.p_tokenmask
        p_constmask = self.hparams.p_constmask / p_sum
        p_srlmask = self.hparams.p_srlmask / p_sum
        p_wordmask = self.hparams.p_wordmask / p_sum
        p_tokenmask = self.hparams.p_tokenmask / p_sum

        mask_type = ''

        # if self.hparams.model =="xlnet":
        #     return self.random_xlnet_token(tokens, tokenizer), "last token"
        if prob < p_constmask:
            mask_type = "const mask"
            return  self.random_span(tokens, tokenizer, const_list, mask_pb = 0.15), mask_type
        prob -= p_constmask
        if prob < p_srlmask:
            mask_type = "srl mask"
            return self.random_span(tokens, tokenizer, srl_list, mask_pb = 0.15), mask_type
        prob -= p_srlmask
        if prob < p_wordmask:
            mask_type = "word mask"
            return self.random_span(tokens, tokenizer, word_list, mask_pb = 0.15), mask_type
        mask_type = "tokens mask"
        return self.random_token(tokens, tokenizer), mask_type


    def random_xlnet_token(self, tokens, tokenizer):

        output_label = []
        output_token = []
        mask_label = []

        for i, token in enumerate(tokens):

            prob = random.random()
            # mask token with 15% probability
            if prob < 0.15:
                prob /= 0.15

                # 80% randomly change token to mask token
                if prob < 0.8:
                    tokens[i] = tokenizer.mask_token

                # 10% randomly change token to random token
                elif prob < 0.9:
                    tokens[i] = tokens[i] if self.hparams.model =="xlnet" else random.choice(list(self.vocab.items()))[0]

                # -> rest 10% randomly keep current token

                # append current token to output (we will predict these later)
                try:
                    output_label.append(self.vocab[token])
                except KeyError:
                    # For unknown words (should not occur with BPE vocab)
                    output_label.append(self.vocab[tokenizer.unk_token])
                    logger.warning("Cannot find token '{}' in vocab. Using [UNK] insetad".format(token))
                output_token.append(token)
                mask_label.append(1)
            else:
                # no masking token (will be ignored by loss function later)
                output_label.append(-1)
                output_token.append(-1)
                mask_label.append(0)

        return tokens, output_label, output_token, mask_label

    def random_span(self, tokens, tokenizer, span_list, mask_pb):

        output_label = [-1 for _ in tokens]# no masking token (will be ignored by loss function later)
        output_token = [" " for _ in tokens]
        mask_label = [0 for _ in tokens]

        if self.hparams.use_alltoken:
            for i, token in enumerate(tokens):
                try:
                    output_label[i] = self.vocab[token]
                except KeyError:
                    # For unknown words (should not occur with BPE vocab)
                    output_label[i] = self.vocab[tokenizer.unk_token]
                    logger.warning("Cannot find token '{}' in vocab. Using [UNK] insetad".format(tokens[i]))
                mask_label[i] = 1

        # span_list = [(span[0], span[1]) for span in span_list if span[1] - span[0] + 1 <= len(tokens) * 0.15]
        flags = [False for _ in tokens]
        span_list = [(right - left, left, right) for left, right in span_list if right < len(tokens)]
        span_list.sort()

        span_list.reverse()
        # print(span_list)
        for i, span in enumerate(span_list):
            prob = random.random()
            start = span[1]
            end = span[2]
            span_len = end - start + 1
            # print(prob, span_len, mask_pb / span_len)
            if not max(flags[start:end + 1]) and prob < mask_pb / span_len:
                # prob /= mask_pb
                # prob *= span_len
                for j in range(start, end + 1):
                    token = tokens[j]
                    prob = random.random()
                    flags[j] = True
                    if prob < 0.8:
                        tokens[j] = tokenizer.mask_token
                    elif prob < 0.9:
                        tokens[j] = tokens[j] if self.hparams.model =="xlnet" else random.choice(list(self.vocab.items()))[0]
                    # -> rest 10% randomly keep current token

                    # append current token to output (we will predict these later)
                    try:
                        output_label[j]=self.vocab[token]
                    except KeyError:
                        # For unknown words (should not occur with BPE vocab)
                        output_label[j]=self.vocab[tokenizer.unk_token]
                        logger.warning("Cannot find token '{}' in vocab. Using [UNK] insetad".format(tokens[j]))
                    output_token[j] = token
                    mask_label[j] = 1

        return tokens, output_label, output_token, mask_label

    # def random_ngram(self, tokens, tokenizer, word_list):
    #     output_label = []
    #
    #     sum = 0
    #     for k in word_list:
    #         sum += k
    #     for i, token in enumerate(tokens):
    #         prob = random.random()
    #         gram_sum = 0
    #         if prob >= 0.15:
    #             output_label.append(-1)
    #         else:
    #             for k in word_list:
    #                 gram_sum += k
    #                 if prob < 0.15 * gram_sum / sum and i + k < len(tokens):
    #                     prob = random.random()
    #                     for j in range(k):
    #                         if prob < 0.8:
    #                             tokens[i + j] = "[MASK]"
    #                         elif prob < 0.9:
    #                             tokens[i + j] = random.choice(list(tokenizer.vocab.items()))[0]
    #                         # -> rest 10% randomly keep current token
    #
    #                         # append current token to output (we will predict these later)
    #                         try:
    #                             output_label.append(tokenizer.vocab[tokens[i + j]])
    #                         except KeyError:
    #                             # For unknown words (should not occur with BPE vocab)
    #                             output_label.append(tokenizer.vocab["[UNK]"])
    #                             logger.warning(
    #                                 "Cannot find token '{}' in vocab. Using [UNK] insetad".format(tokens[i + j]))
    #                 i += j
    #                 break
    #
    #     return tokens, output_label

    def random_token(self, tokens, tokenizer):
        """
        Masking some random tokens for Language Model task with probabilities as in the original BERT paper.
        :param tokens: list of str, tokenized sentence.
        :param tokenizer: Tokenizer, object used for tokenization (we need it's vocab here)
        :return: (list of str, list of int), masked tokens and related labels for LM prediction
        """
        output_label = []
        output_token = []
        mask_label = []

        for i, token in enumerate(tokens):
            prob = random.random()
            # mask token with 15% probability
            if prob < 0.15:
                prob /= 0.15

                # 80% randomly change token to mask token
                if prob < 0.8:
                    tokens[i] = tokenizer.mask_token

                # 10% randomly change token to random token
                elif prob < 0.9:
                    tokens[i] = tokens[i] if self.hparams.model =="xlnet" else random.choice(list(self.vocab.items()))[0]

                # -> rest 10% randomly keep current token

                # append current token to output (we will predict these later)
                try:
                    output_label.append(self.vocab[token])
                except KeyError:
                    # For unknown words (should not occur with BPE vocab)
                    output_label.append(self.vocab[tokenizer.unk_token])
                    logger.warning("Cannot find token '{}' in vocab. Using [UNK] insetad".format(token))
                output_token.append(token)
                mask_label.append(1)
            else:
                # no masking token (will be ignored by loss function later)
                output_label.append(-1)
                output_token.append(-1)
                mask_label.append(0)

        return tokens, output_label, output_token, mask_label

    def convert_example_to_features(self, example, word_start_mask, word_end_mask, max_seq_length, tokenizer, is_ptb):
        """
        Convert a raw sample (pair of sentences as tokenized strings) into a proper training sample with
        IDs, LM labels, input_mask, CLS and SEP tokens etc.
        :param example: InputExample, containing sentence input as strings and is_next label
        :param max_seq_length: int, maximum length of sequence.
        :param tokenizer: Tokenizer
        :return: InputFeatures, containing all inputs and labels of one sample as IDs (as used for model training)
        `cls_token_at_end` define the location of the CLS token:
            - False (Default, BERT/XLM pattern): [CLS] + A + [SEP] + B + [SEP]
            - True (XLNet/GPT pattern): A + [SEP] + B + [SEP] + [CLS]
        `cls_token_segment_id` define the segment id associated to the CLS token (0 for BERT, 2 for XLNet)

        XLNet Examples::

        tokenizer = XLNetTokenizer.from_pretrained('xlnet-large-cased')
        model = XLNetLMHeadModel.from_pretrained('xlnet-large-cased')
        # We show how to setup inputs to predict a next token using a bi-directional context.
        input_ids = torch.tensor(tokenizer.encode("Hello, my dog is very <mask>")).unsqueeze(0)  # We will predict the masked token
        perm_mask = torch.zeros((1, input_ids.shape[1], input_ids.shape[1]), dtype=torch.float)
        perm_mask[:, :, -1] = 1.0  # Previous tokens don't see last token
        target_mapping = torch.zeros((1, 1, input_ids.shape[1]), dtype=torch.float)  # Shape [1, 1, seq_length] => let's predict one token
        target_mapping[0, 0, -1] = 1.0  # Our first (and only) prediction will be the last token of the sequence (the masked token)
        outputs = model(input_ids, perm_mask=perm_mask, target_mapping=target_mapping)
        next_token_logits = outputs[0]  # Output has shape [target_mapping.size(0), target_mapping.size(1), config.vocab_size]

        """

        tokens_a = example.tokens_a
        origin_a = copy.deepcopy(example.tokens_a)
        tokens_b = example.tokens_b
        orgin_b = copy.deepcopy(example.tokens_b)
        word_a = example.word_a
        word_b = example.word_b
        const_a = example.const_a
        const_b = example.const_b
        srl_a = example.srl_a
        srl_b = example.srl_b
        # Modifies `tokens_a` and `tokens_b` in place so that the total
        # length is less than the specified length.
        # Account for [CLS], [SEP], [SEP] with "- 3"
        # assert len(tokens_a) + len(tokens_b) <= max_seq_length - 3
        if is_ptb:

            (t1_random, t1_label, t1_label_token, mask_label_t1), mask_type_1 = self.random_mask(tokens_a, word_a, const_a, srl_a, tokenizer)
            if self.hparams.model =="xlnet":
                lm_label_ids = (t1_label + [-1] + [-1])
                lm_label_mask = ([0] + mask_label_t1 + [0])
            else:
                lm_label_ids = ([-1] + t1_label + [-1])
                lm_label_mask = ([0] + mask_label_t1 + [0])
            mask_type_2 = ""
        else:

            _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)

            (t1_random, t1_label, t1_label_token, mask_label_t1), mask_type_1 = self.random_mask(tokens_a, word_a, const_a, srl_a, tokenizer)
            (t2_random, t2_label, t2_label_token, mask_label_t2), mask_type_2 = self.random_mask(tokens_b, word_b, const_b, srl_b, tokenizer)
            # concatenate lm labels and account for CLS, SEP, SEP
            if self.hparams.model =="xlnet":
                lm_label_ids = (t2_label + [-1] + t1_label + [-1] + [-1])
                lm_label_mask = ([0] + mask_label_t1 + [0] + mask_label_t2 + [0])
            else:
                lm_label_ids = ([-1] + t1_label + [-1] + t2_label + [-1])
                lm_label_mask = ([0] + mask_label_t1 + [0] + mask_label_t2 + [0])

        # The convention in BERT is:
        # (a) For sequence pairs:
        #  tokens:         [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
        #  type_ids:        0   0  0    0    0     0       0 0    1  1  1  1   1 1
        #  word_start_mask: 1   1   1   1   0       0       1 1   0 0  0   0   0  0 0
        # (b) For single sequences:
        #  tokens:   [CLS] the dog is hairy . [SEP]
        #  type_ids: 0   0   0   0  0     0 0
        #
        # Where "type_ids" are used to indicate whether this is the first
        # sequence or the second sequence. The embedding vectors for `type=0` and
        # `type=1` were learned during pre-training and are added to the wordpiece
        # embedding vector (and position vector). This is not *strictly* necessary
        # since the [SEP] token unambigiously separates the sequences, but it makes
        # it easier for the model to learn the concept of sequences.
        #
        # For classification tasks, the first vector (corresponding to [CLS]) is
        # used as as the "sentence vector". Note that this only makes sense because
        # the entire model is fine-tuned.


        tokens = []
        origin_tokens = []
        segment_ids = []
        if not self.hparams.model =="xlnet":
            tokens.append(tokenizer.cls_token)
            origin_tokens.append(tokenizer.cls_token)
            segment_ids.append(0)

        for token in tokens_a:
            tokens.append(token)
            segment_ids.append(1 if self.hparams.model =="xlnet" and not is_ptb else 0)

        origin_tokens.extend(origin_a)

        tokens.append(tokenizer.sep_token)
        origin_tokens.append(tokenizer.sep_token)
        segment_ids.append(1 if self.hparams.model =="xlnet" and not is_ptb else 0)

        if self.hparams.model =="xlnet":
            tokens.append(tokenizer.cls_token)
            origin_tokens.append(tokenizer.cls_token)
            segment_ids.append(2)

        if not is_ptb:
            assert len(tokens_b) > 0
            tokens_b_list = []
            segment_ids_b = []
            word_start_mask_b = []
            word_end_mask_b = []
            for token in tokens_b:
                tokens_b_list.append(token)
                segment_ids_b.append(0 if self.hparams.model =="xlnet"and not is_ptb else 1)
                word_start_mask_b.append(0)
                word_end_mask_b.append(0)
            origin_tokens.extend(orgin_b)

            tokens_b_list.append(tokenizer.sep_token)
            origin_tokens.append(tokenizer.sep_token)
            segment_ids_b.append(0 if self.hparams.model =="xlnet" and not is_ptb else 1)
            word_start_mask_b.append(0)   #second tokens word mask is 0
            word_end_mask_b.append(0)

            if self.hparams.model =="xlnet":
                tokens = tokens_b_list + tokens
                segment_ids = segment_ids_b + segment_ids
                word_start_mask = word_start_mask_b + word_start_mask
                word_end_mask = word_end_mask_b + word_end_mask
            else:
                tokens = tokens + tokens_b_list
                segment_ids = segment_ids + segment_ids_b
                word_start_mask = word_start_mask + word_start_mask_b
                word_end_mask = word_end_mask + word_end_mask_b


        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        origin_ids = tokenizer.convert_tokens_to_ids(origin_tokens)
        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding_length = max_seq_length - len(input_ids)
        if self.hparams.model =="xlnet":
            input_ids = ([0] * padding_length) + input_ids
            input_mask = ([0] * padding_length) + input_mask
            segment_ids = ([4] * padding_length) + segment_ids
            lm_label_ids = ([-1] * padding_length) + lm_label_ids
            word_start_mask = ([0] * padding_length) + word_start_mask
            word_end_mask = ([0] * padding_length) + word_end_mask

            perm_mask = torch.zeros((len(input_ids.shape), len(input_ids.shape)), dtype=torch.float)
            num_lm = 0
            target_mapping = []
            for i, lm_label in enumerate(lm_label_ids):
                if lm_label != -1:
                    perm_mask[:, i] = 1.0  # Previous tokens don't see last token
                    target_mapping.append([0.0 if i != j else 1.0 for j, _ in enumerate(lm_label_ids)])

            # target_mapping = torch.zeros((num_lm, len(input_ids.shape)),
            #                              dtype=torch.float)  # Shape [1, 1, seq_length] => let's predict one token
            # target_mapping[0, -1] = 1.0  # Our first (and only) prediction will be the last token of the sequence (the masked token)
        else:
            input_ids = input_ids + ([0] * padding_length)
            origin_ids = origin_ids + ([0] * padding_length)
            input_mask = input_mask + ([0] * padding_length)
            segment_ids = segment_ids + ([0] * padding_length)
            lm_label_ids = lm_label_ids + ([-1] * padding_length)
            lm_label_mask = lm_label_mask + ([0] * padding_length)
            word_start_mask = word_start_mask + ([0] * padding_length)
            word_end_mask = word_end_mask + ([0] * padding_length)

            perm_mask = []
            target_mapping = []


        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length
        assert len(lm_label_ids) == max_seq_length
        assert len(word_start_mask) == max_seq_length
        assert len(word_end_mask) == max_seq_length

        if example.guid < 10:
            logger.info("*** Example ***")
            logger.info("guid: %s" % (example.guid))
            logger.info("tokens: %s" % " ".join(
                [str(x) for x in tokens]))
            logger.info("origin_tokens: %s" % " ".join(
                [str(x) for x in origin_tokens]))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("origin_ids: %s" % " ".join([str(x) for x in origin_ids]))
            logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
            logger.info("word_start_mask: %s" % " ".join([str(x) for x in word_start_mask]))
            logger.info("word_end_mask: %s" % " ".join([str(x) for x in word_end_mask]))
            logger.info(
                "segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
            logger.info("LM label: %s " % (lm_label_ids))
            logger.info("LM label mask: %s " % (lm_label_mask))
            logger.info("LM label token: %s " % (t1_label_token))
            logger.info("Is next sentence label: %s " % (example.is_next))
            logger.info("Is PTB dataset: %s " % (is_ptb))
            logger.info("Mask Tpye: %s " % ([mask_type_1, mask_type_2]))

        features = InputFeatures(input_ids=input_ids,
                                 origin_ids = origin_ids,
                                 input_mask=input_mask,
                                 word_start_mask = word_start_mask,
                                 word_end_mask = word_end_mask,
                                 segment_ids=segment_ids,
                                 perm_mask = perm_mask,
                                 target_mapping = target_mapping,
                                 lm_label_ids=lm_label_ids,
                                 lm_label_mask = lm_label_mask,
                                 is_next=example.is_next)
        return features


class InputExample(object):
    """A single training/test example for the language model."""

    def __init__(self, guid, tokens_a, word_a, const_a, srl_a, tokens_b=None, word_b = None, const_b = None, srl_b = None, is_next=None, lm_labels=None):
        """Constructs a InputExample.

        Args:
            guid: Unique id for the example.
            tokens_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            tokens_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.tokens_a = tokens_a
        self.tokens_b = tokens_b
        self.word_a = word_a
        self.word_b = word_b
        self.const_a = const_a
        self.srl_a = srl_a
        self.const_b = const_b
        self.srl_b = srl_b
        self.is_next = is_next  # nextSentence
        self.lm_labels = lm_labels  # masked words for language model


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, origin_ids, input_mask, word_start_mask, word_end_mask, segment_ids, perm_mask,
                 target_mapping, is_next, lm_label_ids, lm_label_mask):
        self.input_ids = input_ids
        self.origin_ids = origin_ids
        self.input_mask = input_mask
        self.word_start_mask = word_start_mask
        self.word_end_mask = word_end_mask
        self.segment_ids = segment_ids
        self.perm_mask = perm_mask
        self.target_mapping = target_mapping
        self.is_next = is_next
        self.lm_label_ids = lm_label_ids
        self.lm_label_mask=lm_label_mask


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while len(tokens_b) > 0 :
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        tokens_b.pop()
    assert  len(tokens_a) + len(tokens_b) <= max_length
        # if len(tokens_a) > len(tokens_b):
        #     tokens_a.pop()
        # else:
        #     tokens_b.pop()
