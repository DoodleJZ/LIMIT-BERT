
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
class EVALDataset(Dataset):
    def __init__(self, hparams, ptb_dataset, dataset_name, tokenizer, seq_len, encoding="utf-8"):
        if hparams.model =="xlnet":
            self.vocab = tokenizer.sp_model
        else:
            self.vocab = tokenizer.vocab
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.encoding = encoding
        self.current_doc = 0  # to avoid random sentence from same doc
        self.ptb_dataset = ptb_dataset.ptb_dataset
        self.dataset = ptb_dataset.ptb_dataset[dataset_name]
        self.sample_counter = 0
        self.num_data = len(self.dataset)

        self.hparams = hparams
        if hparams.bert_transliterate:
            from transliterate import TRANSLITERATIONS
            self.bert_transliterate = TRANSLITERATIONS[hparams.bert_transliterate]
        else:
            self.bert_transliterate = None

    def __len__(self):
        # last line of doc won't be used, because there's no "nextSentence". Additionally, we start counting at 0.
        return self.num_data

    def __getitem__(self, item):
        cur_id = self.sample_counter
        self.sample_counter += 1

        # t1, t2, is_next_label = self.random_sent(item)

        data_sent = self.dataset[item]

        is_next_label = -1


        # tokenize
        # tokens_a = self.tokenizer.tokenize(t1[0])
        # tokens_b = self.tokenizer.tokenize(t2[0])

        tokens_a, word_start_mask, word_end_mask = self.token_span(data_sent)

        # combine to one sample
        cur_example = InputExample(guid=cur_id, tokens_a=tokens_a, is_next=is_next_label)

        # transform sample to features
        cur_features = self.convert_example_to_features(cur_example, word_start_mask, word_end_mask, self.seq_len, self.tokenizer)

        cur_data = (torch.tensor(cur_features.input_ids),
                    torch.tensor(cur_features.input_mask),
                    torch.tensor(cur_features.word_start_mask),
                    torch.tensor(cur_features.word_end_mask),
                    torch.tensor(cur_features.segment_ids),
                    torch.tensor(cur_features.lm_label_ids),
                    torch.tensor(cur_features.is_next),
                    json.dumps(data_sent),
                    )

        return cur_data

    def token_span(self, sent):

        tokens = []
        word_start_mask = []
        word_end_mask = []
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
            for _ in range(len(word_tokens)):
                word_start_mask.append(0)
                word_end_mask.append(0)
                idx += 1
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

        return tokens, word_start_mask, word_end_mask


    def convert_example_to_features(self, example, word_start_mask, word_end_mask, max_seq_length, tokenizer):
        """
        Convert a raw sample (pair of sentences as tokenized strings) into a proper training sample with
        IDs, LM labels, input_mask, CLS and SEP tokens etc.
        :param example: InputExample, containing sentence input as strings and is_next label
        :param max_seq_length: int, maximum length of sequence.
        :param tokenizer: Tokenizer
        :return: InputFeatures, containing all inputs and labels of one sample as IDs (as used for model training)
        """
        tokens_a = example.tokens_a
        # Modifies `tokens_a` and `tokens_b` in place so that the total
        # length is less than the specified length.
        # Account for [CLS], [SEP], [SEP] with "- 3"
        assert len(tokens_a) <= max_seq_length - 2
        # _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)

        # t1_random, t1_label = self.random_mask(tokens_a, word_a, const_a, srl_a, tokenizer)
        # t2_random, t2_label = self.random_mask(tokens_b, word_b, const_b, srl_b, tokenizer)
        # concatenate lm labels and account for CLS, SEP, SEP

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
        segment_ids = []
        if not self.hparams.model =="xlnet":
            tokens.append(tokenizer.cls_token)
            segment_ids.append(0)

        for token in tokens_a:
            tokens.append(token)
            segment_ids.append(0)

        tokens.append(tokenizer.sep_token)
        segment_ids.append(0)

        if self.hparams.model =="xlnet":
            tokens.append(tokenizer.cls_token)
            segment_ids.append(2)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)
        lm_label_ids = [-1] * max_seq_length

        # Zero-pad up to the sequence length.
        padding_length = max_seq_length - len(input_ids)
        if self.hparams.model =="xlnet":
            input_ids = ([0] * padding_length) + input_ids
            input_mask = ([0] * padding_length) + input_mask
            segment_ids = ([4] * padding_length) + segment_ids
            word_start_mask = ([0] * padding_length) + word_start_mask
            word_end_mask = ([0] * padding_length) + word_end_mask
        else:
            input_ids = input_ids + ([0] * padding_length)
            input_mask = input_mask + ([0] * padding_length)
            segment_ids = segment_ids + ([0] * padding_length)
            word_start_mask = word_start_mask + ([0] * padding_length)
            word_end_mask = word_end_mask + ([0] * padding_length)

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length
        assert len(lm_label_ids) == max_seq_length
        assert len(word_start_mask) == max_seq_length
        assert len(word_end_mask) == max_seq_length

        # # Zero-pad up to the sequence length.
        # while len(input_ids) < max_seq_length:
        #     input_ids.append(0)
        #     input_mask.append(0)
        #     segment_ids.append(0)
        #     lm_label_ids.append(-1)
        #     word_start_mask.append(0)
        #     word_end_mask.append(0)
        #
        # assert len(input_ids) == max_seq_length
        # assert len(input_mask) == max_seq_length
        # assert len(segment_ids) == max_seq_length
        # assert len(lm_label_ids) == max_seq_length
        # assert len(word_start_mask) == max_seq_length
        # assert len(word_end_mask) == max_seq_length

        # if example.guid < 5:
        #     logger.info("*** Example ***")
        #     logger.info("guid: %s" % (example.guid))
        #     logger.info("tokens: %s" % " ".join(
        #         [str(x) for x in tokens]))
        #     logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
        #     logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
        #     logger.info("word_start_mask: %s" % " ".join([str(x) for x in word_start_mask]))
        #     logger.info("word_end_mask: %s" % " ".join([str(x) for x in word_end_mask]))
        #     logger.info(
        #         "segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
        #     logger.info("LM label: %s " % (lm_label_ids))
        #     logger.info("Is next sentence label: %s " % (example.is_next))

        features = InputFeatures(input_ids=input_ids,
                                 input_mask=input_mask,
                                 word_start_mask = word_start_mask,
                                 word_end_mask = word_end_mask,
                                 segment_ids=segment_ids,
                                 lm_label_ids=lm_label_ids,
                                 is_next=example.is_next)
        return features


class InputExample(object):
    """A single training/test example for the language model."""

    def __init__(self, guid, tokens_a, is_next=None, lm_labels=None):
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
        self.is_next = is_next  # nextSentence
        self.lm_labels = lm_labels  # masked words for language model


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, word_start_mask, word_end_mask, segment_ids, is_next, lm_label_ids):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.word_start_mask = word_start_mask
        self.word_end_mask = word_end_mask
        self.segment_ids = segment_ids
        self.is_next = is_next
        self.lm_label_ids = lm_label_ids
