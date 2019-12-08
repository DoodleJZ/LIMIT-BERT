from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import logging
import time
import argparse
from tqdm import tqdm, trange

import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler

from pretrained_bert.tokenization import BertTokenizer
from pretrained_bert.modeling import BertForPreTraining
from pretrained_bert.optimization import BertAdam


from torch.utils.data import Dataset
import random
from bert_dataset import BERTDataset
from eval_dataset import EVALDataset
from ptb_dataset import PTBDataset
from Evaluator import evaluate
from Evaluator import dep_eval
from Evaluator import srl_eval
from Evaluator import pos_eval
import makehp
import json
import Zmodel
import trees
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


def torch_load(load_path):
    if Zmodel.use_cuda:
        return torch.load(load_path)
    else:
        return torch.load(load_path, map_location=lambda storage, location: storage)


def format_elapsed(start_time):
    elapsed_time = int(time.time() - start_time)
    minutes, seconds = divmod(elapsed_time, 60)
    hours, minutes = divmod(minutes, 60)
    days, hours = divmod(hours, 24)
    elapsed_string = "{}h{:02}m{:02}s".format(hours, minutes, seconds)
    if days > 0:
        elapsed_string = "{}d{}".format(days, elapsed_string)
    return elapsed_string

class FScore(object):
    def __init__(self, recall, precision, fscore):
        self.recall = recall
        self.precision = precision
        self.fscore = fscore

    def __str__(self):
        return "(Recall={:.2f}, Precision={:.2f}, FScore={:.2f})".format(
            self.recall, self.precision, self.fscore)

class EvalManyTask(object):
    def __init__(self, device, hparams, ptb_dataset, task_list, bert_tokenizer, seq_len, eval_batch_size, evalb_dir, model_path_base, log_path):


        self.hparams = hparams
        self.device = device
        self.ptb_dataset = ptb_dataset.ptb_dataset
        self.evalb_dir = evalb_dir
        self.model_path_base = model_path_base
        self.test_model_path = None
        self.log_path = log_path
        self.summary_dict = {}
        self.task_datasent = {}
        self.task_dataloader = {}
        self.task_list = task_list
        self.eval_batch_size = eval_batch_size
        self.seq_len = seq_len
        for task_name in task_list:
            self.task_datasent[task_name] = EVALDataset(hparams, ptb_dataset, task_name, bert_tokenizer, seq_len)
            self.task_dataloader[task_name] = DataLoader(self.task_datasent[task_name], sampler=SequentialSampler(self.task_datasent[task_name]),
                                                         batch_size = eval_batch_size)

        self.best_dev_score = -np.inf
        self.best_model_path = None
        self.summary_dict['dev_synconst'] = 0
        self.summary_dict['dev_syndep_uas'] = 0
        self.summary_dict['dev_syndep_las'] = 0
        self.summary_dict['dev_srlspan'] = 0
        self.summary_dict['dev_srldep'] = 0
        self.summary_dict['dev_pos'] = 0

        self.summary_dict['test_synconst'] = 0
        self.summary_dict['test_syndep_uas'] = 0
        self.summary_dict['test_syndep_las'] = 0
        self.summary_dict['test_srlspan'] = 0
        self.summary_dict['test_srldep'] = 0
        self.summary_dict['test_pos'] = 0

        self.summary_dict['brown_srlspan'] = 0
        self.summary_dict['brown_srldep'] = 0


    def eval_multitask(self, model, start_time, epoch_num):

        logger.info("***** Running dev *****")
        logger.info("  Batch size = %d", self.eval_batch_size)
        logger.info("  Seq Len = %d", self.seq_len)

        # assert self.test_model_path is not None
        # info = torch_load(self.test_model_path + '.pt')
        # TESTmodel = Zmodel.Jointmodel.from_spec(info['spec'], info['state_dict'])
        # TESTmodel.to(self.device)
        # TESTmodel.eval()

        dev_start_time = time.time()

        length = 0
        for i in range(7):
            length += 10
            print("Start Length " + str(length) + " Dev Eval:")
            if self.hparams.joint_syn:
                print("===============================================")
                print("Start syntax " + str(length) + " dev eval:")
                self.syn_dev(model, length)

            if self.hparams.joint_srl:
                print("===============================================")
                print("Start srl span " + str(length) + "dev eval:")
                self.srlspan_dev(model, length)

                print("===============================================")
                print("Start srl dep " + str(length) + " dev eval:")
                self.srldep_dev(model, length)
        print("Start Dev Eval:")
        if self.hparams.joint_syn:
            print("===============================================")
            print("Start syntax dev eval:")
            self.syn_dev(model)

        if self.hparams.joint_srl:
            print("===============================================")
            print("Start srl span dev eval:")
            self.srlspan_dev(model)

            print("===============================================")
            print("Start srl dep dev eval:")
            self.srldep_dev(model)


        print(
            "dev-elapsed {} "
            "total-elapsed {}".format(
                format_elapsed(dev_start_time),
                format_elapsed(start_time),
            )
        )

        print(
            '============================================================================================================================')

        print("Start Test Eval:")
        test_start_time = time.time()

        if self.hparams.joint_syn:
            print("===============================================")
            print("Start syntax test eval:")
            self.syn_test(model)

        if self.hparams.joint_srl:
            print("===============================================")
            print("Start srl span test eval:")
            self.srlspan_test(model)

            print("===============================================")
            print("Start srl dep test eval:")
            self.srldep_test(model)

        print(
            "test-elapsed {} "
            "total-elapsed {}".format(
                format_elapsed(test_start_time),
                format_elapsed(start_time),
            )
        )

        return self.best_model_path, False

        self.summary_dict['total dev score'] = self.summary_dict['dev_synconst'].fscore + self.summary_dict['dev_syndep_las'] + \
                                               self.summary_dict['dev_srlspan'].fscore + self.summary_dict['dev_srldep'].fscore + self.summary_dict['dev_pos']

        log_data = "{} epoch , dev-fscore {:},test-fscore {:}, dev-uas {:.2f}, dev-las {:.2f}，" \
                   "test-uas {:.2f}, test-las {:.2f}, dev-srlspan {:}, test-wsj-srlspan {:}, test-brown-srlspan {:}," \
                   " dev-srldep {:},  test-wsj-srldep {:}, test-brown-srldep {:}, dev-pos {:}, test-pos {:}," \
                   "dev_score {:.2f}, best_dev_score {:.2f}" \
            .format(epoch_num, self.summary_dict['dev_synconst'], self.summary_dict['test_synconst'],
                    self.summary_dict['dev_syndep_uas'], self.summary_dict['dev_syndep_las'],
                    self.summary_dict['test_syndep_uas'], self.summary_dict['test_syndep_las'],
                    self.summary_dict['dev_srlspan'], self.summary_dict['test_srlspan'], self.summary_dict['brown_srlspan'],
                    self.summary_dict['dev_srldep'], self.summary_dict['test_srldep'], self.summary_dict['brown_srldep'],
                    self.summary_dict['dev_pos'], self.summary_dict['test_pos'],
                    self.summary_dict['total dev score'], self.best_dev_score)

        if not os.path.exists(self.log_path):
            flog = open(self.log_path, 'w')
        flog = open(self.log_path, 'r+')
        content = flog.read()
        flog.seek(0, 0)
        flog.write(log_data + '\n' + content)

        is_save_model = False

        if self.summary_dict['total dev score'] > self.best_dev_score:
            if self.best_model_path is not None:
                extensions = [".pt"]
                for ext in extensions:
                    path = self.best_model_path + ext
                    if os.path.exists(path):
                        print("Removing previous model file {}...".format(path))
                        os.remove(path)

            self.best_dev_score = self.summary_dict['total dev score']

            self.best_model_path = "{}_best_dev={:.2f}_devuas={:.2f}_devlas={:.2f}_devsrlspan={:.2f}_devsrldep={:.2f}".format(
                self.model_path_base, self.summary_dict['dev_synconst'].fscore, self.summary_dict['dev_syndep_uas'], self.summary_dict['dev_syndep_las'],
                self.summary_dict['dev_srlspan'].fscore, self.summary_dict['dev_srldep'].fscore)
            is_save_model = True

        return self.best_model_path, is_save_model


    def syn_dev(self, model, leng = 0):
        syntree_pred = []

        assert 'dev_synconst' in self.task_list

        if leng == 0:
            str_leng = ""
        else:
            str_leng = str(leng)
        dev_pred_head = []
        dev_pred_type = []
        for step, batch in enumerate(tqdm(self.task_dataloader['dev_synconst' + str_leng], desc="Syntax Dev")):
            input_ids, input_mask, word_start_mask, word_end_mask, segment_ids, lm_label_ids, is_next, sent = batch
            dis_idx = [i for i in range(len(input_ids))]
            dis_idx = torch.tensor(dis_idx)
            batch = dis_idx, input_ids, input_mask, word_start_mask, word_end_mask, segment_ids, lm_label_ids, is_next
            bert_data = tuple(t.to(self.device) for t in batch)
            sentences = [json.loads(sent_str) for sent_str in sent]
            # linz, head, type, _, _, _ = model(sentences=sentences, bert_data=bert_data)
            # dev_pred_head.extend([json.loads(head_str) for head_str in head])
            # dev_pred_type.extend([json.loads(type_str) for type_str in type])
            # syntree_pred.extend(linz)
            syntree, _, _ = model(sentences=sentences, bert_data=bert_data)
            syntree_pred.extend(syntree)

        # const parsing:
        self.summary_dict['dev_synconst'+str_leng] = evaluate.evalb(self.evalb_dir, self.ptb_dataset['dev_synconst_tree'+str_leng], syntree_pred)

        # dep parsing:

        dev_pred_head = [[leaf.father for leaf in tree.leaves()] for tree in syntree_pred]
        dev_pred_type = [[leaf.type for leaf in tree.leaves()] for tree in syntree_pred]
        syndep_dev_pos = [[leaf.tag for leaf in tree.leaves()] for tree in self.ptb_dataset['dev_synconst_tree'+str_leng]]
        assert len(dev_pred_head) == len(dev_pred_type)
        assert len(dev_pred_type) == len(self.ptb_dataset['dev_syndep_type'+str_leng])

        self.summary_dict['dev_syndep_uas'+str_leng], self.summary_dict['dev_syndep_las'+str_leng] = \
            dep_eval.eval(len(dev_pred_head), self.ptb_dataset['dev_syndep_sent'+str_leng], syndep_dev_pos,
                          dev_pred_head, dev_pred_type, self.ptb_dataset['dev_syndep_head'+str_leng], self.ptb_dataset['dev_syndep_type'+str_leng],
                          punct_set=self.hparams.punctuation, symbolic_root=False)

    def srlspan_dev(self, model, leng= 0):
        srlspan_pred = []

        assert 'dev_srlspan' in self.task_list
        if leng == 0:
            str_leng = ""
        else:
            str_leng = str(leng)
        pos_pred = []
        for step, batch in enumerate(tqdm(self.task_dataloader['dev_srlspan'+str_leng], desc="Srlspan Dev")):
            input_ids, input_mask, word_start_mask, word_end_mask, segment_ids, lm_label_ids, is_next, sent = batch
            dis_idx = [i for i in range(len(input_ids))]
            dis_idx = torch.tensor(dis_idx)
            batch = dis_idx, input_ids, input_mask, word_start_mask, word_end_mask, segment_ids, lm_label_ids, is_next
            bert_data = tuple(t.to(self.device) for t in batch)
            sentences = [json.loads(sent_str) for sent_str in sent]
            # _, _, _, srlspan_list, _, pos_list = model(sentences=sentences, bert_data=bert_data)
            # for srlspan_str in srlspan_list:
            #
            #     srlspan = {}
            #     srlspan_dict = json.loads(srlspan_str)
            #     for pred_id, argus in srlspan_dict.items():
            #         srlspan[int(pred_id)] = [(int(a[0]), int(a[1]), a[2]) for a in argus]
            #
            #     srlspan_pred.append(srlspan)

            # pos_pred.extend([json.loads(pos) for pos in pos_list])
            srlspan_tree, srlspan_dict, _ = model(sentences=sentences, bert_data=bert_data)
            srlspan_pred.extend(srlspan_dict)

            pos_pred.extend([[leaf.goldtag for leaf in tree.leaves()] for tree in srlspan_tree])

        precision, recall, f1, ul_prec, ul_recall, ul_f1 = (
            srl_eval.compute_srl_f1(self.ptb_dataset['dev_srlspan_sent'+str_leng], self.ptb_dataset['dev_srlspan_dict'+str_leng], srlspan_pred,
                                    srl_conll_eval_path=False))
        self.summary_dict['dev_srlspan'+str_leng] = FScore(recall, precision, f1)

        print("===============================================")
        print("Start Pos dev eval:")

        self.summary_dict['dev_pos'+str_leng] = pos_eval.eval(self.ptb_dataset['dev_srlspan_goldpos'+str_leng], pos_pred)

    def srldep_dev(self, model, leng= 0):

        assert 'dev_srldep' in self.task_list
        if leng == 0:
            str_leng = ""
        else:
            str_leng = str(leng)
        srldep_pred = []
        for step, batch in enumerate(tqdm(self.task_dataloader['dev_srldep'+str_leng], desc="Srldep Dev")):
            input_ids, input_mask, word_start_mask, word_end_mask, segment_ids, lm_label_ids, is_next, sent = batch
            dis_idx = [i for i in range(len(input_ids))]
            dis_idx = torch.tensor(dis_idx)
            batch = dis_idx, input_ids, input_mask, word_start_mask, word_end_mask, segment_ids, lm_label_ids, is_next
            bert_data = tuple(t.to(self.device) for t in batch)
            sentences = [json.loads(sent_str) for sent_str in sent]
            # _, _, _, _, srldep_list, _ = model(sentences=sentences, bert_data=bert_data)
            # for srldep_str in srldep_list:
            #
            #     srldep = {}
            #     srldep_dict = json.loads(srldep_str)
            #     for pred_id, argus in srldep_dict.items():
            #         srldep[int(pred_id)] = [(int(a[0]), a[2]) for a in argus]
            #
            #     srldep_pred.append(srldep)

            _, _, srldepdict = model(sentences=sentences, bert_data=bert_data)
            srldep_pred.extend(srldepdict)


        precision, recall, f1 = (
            srl_eval.compute_dependency_f1(self.ptb_dataset['dev_srldep_sent'+str_leng], self.ptb_dataset['dev_srldep_dict'+str_leng], srldep_pred,
                                           srl_conll_eval_path=False, use_gold=False))

        self.summary_dict['dev_srldep'+str_leng] = FScore(recall, precision, f1)

    def syn_test(self, model):
        syntree_pred = []
        dev_pred_head = []
        dev_pred_type = []
        assert 'test_synconst' in self.task_list

        for step, batch in enumerate(tqdm(self.task_dataloader['test_synconst'], desc="Syntax Test")):
            input_ids, input_mask, word_start_mask, word_end_mask, segment_ids, lm_label_ids, is_next, sent = batch
            dis_idx = [i for i in range(len(input_ids))]
            dis_idx = torch.tensor(dis_idx)
            batch = dis_idx, input_ids, input_mask, word_start_mask, word_end_mask, segment_ids, lm_label_ids, is_next
            bert_data = tuple(t.to(self.device) for t in batch)
            sentences = [json.loads(sent_str) for sent_str in sent]
            # linz, head, type, _, _, _ = model(sentences=sentences, bert_data=bert_data)
            # dev_pred_head.extend([json.loads(head_str) for head_str in head])
            # dev_pred_type.extend([json.loads(type_str) for type_str in type])
            # syntree_pred.extend(linz)
            syntree, _, _ = model(sentences=sentences, bert_data=bert_data)
            syntree_pred.extend(syntree)

        # const parsing:
        self.summary_dict['test_synconst'] = evaluate.evalb(self.evalb_dir, self.ptb_dataset['test_synconst_tree'], syntree_pred)

        # dep parsing:

        dev_pred_head = [[leaf.father for leaf in tree.leaves()] for tree in syntree_pred]
        dev_pred_type = [[leaf.type for leaf in tree.leaves()] for tree in syntree_pred]
        syndep_dev_pos = [[leaf.tag for leaf in tree.leaves()] for tree in self.ptb_dataset['test_synconst_tree']]
        assert len(dev_pred_head) == len(dev_pred_type)
        assert len(dev_pred_type) == len(self.ptb_dataset['test_syndep_type'])

        self.summary_dict['test_syndep_uas'], self.summary_dict['test_syndep_las'] = \
            dep_eval.eval(len(dev_pred_head), self.ptb_dataset['test_syndep_sent'], syndep_dev_pos,
                          dev_pred_head, dev_pred_type, self.ptb_dataset['test_syndep_head'],
                          self.ptb_dataset['test_syndep_type'],
                          punct_set=self.hparams.punctuation, symbolic_root=False)

    def srlspan_test(self, model):
        srlspan_pred = []

        assert 'test_srlspan' in self.task_list

        pos_pred = []
        for step, batch in enumerate(tqdm(self.task_dataloader['test_srlspan'], desc="Srlspan Test")):
            input_ids, input_mask, word_start_mask, word_end_mask, segment_ids, lm_label_ids, is_next, sent = batch
            dis_idx = [i for i in range(len(input_ids))]
            dis_idx = torch.tensor(dis_idx)
            batch = dis_idx, input_ids, input_mask, word_start_mask, word_end_mask, segment_ids, lm_label_ids, is_next
            bert_data = tuple(t.to(self.device) for t in batch)
            sentences = [json.loads(sent_str) for sent_str in sent]
            # _, _, _, srlspan_list, _, pos_list = model(sentences=sentences, bert_data=bert_data)
            # for srlspan_str in srlspan_list:
            #
            #     srlspan = {}
            #     srlspan_dict = json.loads(srlspan_str)
            #     for pred_id, argus in srlspan_dict.items():
            #         srlspan[int(pred_id)] = [(int(a[0]), int(a[1]), a[2]) for a in argus]
            #
            #     srlspan_pred.append(srlspan)
            #
            # pos_pred.extend([json.loads(pos) for pos in pos_list])
            srlspan_tree, srlspan_dict, _ = model(sentences=sentences, bert_data=bert_data)
            srlspan_pred.extend(srlspan_dict)

            pos_pred.extend([[leaf.goldtag for leaf in tree.leaves()] for tree in srlspan_tree])

        precision, recall, f1, ul_prec, ul_recall, ul_f1 = (
            srl_eval.compute_srl_f1(self.ptb_dataset['test_srlspan_sent'], self.ptb_dataset['test_srlspan_dict'],
                                    srlspan_pred,
                                    srl_conll_eval_path=False))
        self.summary_dict['test_srlspan'] = FScore(recall, precision, f1)

        print("===============================================")
        print("Start Pos Test eval:")

        self.summary_dict['test_pos'] = pos_eval.eval(self.ptb_dataset['test_srlspan_goldpos'], pos_pred)

        srlspan_pred = []
        for step, batch in enumerate(tqdm(self.task_dataloader['brown_srlspan'], desc="Srlspan Brown")):
            input_ids, input_mask, word_start_mask, word_end_mask, segment_ids, lm_label_ids, is_next, sent = batch
            dis_idx = [i for i in range(len(input_ids))]
            dis_idx = torch.tensor(dis_idx)
            batch = dis_idx, input_ids, input_mask, word_start_mask, word_end_mask, segment_ids, lm_label_ids, is_next
            bert_data = tuple(t.to(self.device) for t in batch)
            sentences = [json.loads(sent_str) for sent_str in sent]
            # _, _, _, srlspan_list, _, pos_list = model(sentences=sentences, bert_data=bert_data)
            # for srlspan_str in srlspan_list:
            #
            #     srlspan = {}
            #     srlspan_dict = json.loads(srlspan_str)
            #     for pred_id, argus in srlspan_dict.items():
            #         srlspan[int(pred_id)] = [(int(a[0]), int(a[1]), a[2]) for a in argus]
            #
            #     srlspan_pred.append(srlspan)
            _, srlspan_dict, _ = model(sentences=sentences, bert_data=bert_data)
            srlspan_pred.extend(srlspan_dict)


        precision, recall, f1, ul_prec, ul_recall, ul_f1 = (
            srl_eval.compute_srl_f1(self.ptb_dataset['brown_srlspan_sent'], self.ptb_dataset['brown_srlspan_dict'],
                                    srlspan_pred,
                                    srl_conll_eval_path=False))
        self.summary_dict['brown_srlspan'] = FScore(recall, precision, f1)

    def srldep_test(self, model):

        assert 'test_srldep' in self.task_list

        srldep_pred = []
        for step, batch in enumerate(tqdm(self.task_dataloader['test_srldep'], desc="Srldep Test")):
            input_ids, input_mask, word_start_mask, word_end_mask, segment_ids, lm_label_ids, is_next, sent = batch
            dis_idx = [i for i in range(len(input_ids))]
            dis_idx = torch.tensor(dis_idx)
            batch = dis_idx, input_ids, input_mask, word_start_mask, word_end_mask, segment_ids, lm_label_ids, is_next
            bert_data = tuple(t.to(self.device) for t in batch)
            sentences = [json.loads(sent_str) for sent_str in sent]
            # _, _, _, _, srldep_list, _ = model(sentences=sentences, bert_data=bert_data)
            # for srldep_str in srldep_list:
            #
            #     srldep = {}
            #     srldep_dict = json.loads(srldep_str)
            #     for pred_id, argus in srldep_dict.items():
            #         srldep[int(pred_id)] = [(int(a[0]), a[2]) for a in argus]
            #
            #     srldep_pred.append(srldep)
            _, _, srldepdict = model(sentences=sentences, bert_data=bert_data)
            srldep_pred.extend(srldepdict)

        precision, recall, f1 = (
            srl_eval.compute_dependency_f1(self.ptb_dataset['test_srldep_sent'], self.ptb_dataset['test_srldep_dict'],
                                           srldep_pred,
                                           srl_conll_eval_path=False, use_gold=False))

        self.summary_dict['test_srldep'] = FScore(recall, precision, f1)

        srldep_pred = []
        for step, batch in enumerate(tqdm(self.task_dataloader['brown_srldep'], desc="Srldep Brown")):
            input_ids, input_mask, word_start_mask, word_end_mask, segment_ids, lm_label_ids, is_next, sent = batch
            dis_idx = [i for i in range(len(input_ids))]
            dis_idx = torch.tensor(dis_idx)
            batch = dis_idx, input_ids, input_mask, word_start_mask, word_end_mask, segment_ids, lm_label_ids, is_next
            bert_data = tuple(t.to(self.device) for t in batch)
            sentences = [json.loads(sent_str) for sent_str in sent]
            # _, _, _, _, srldep_list, _ = model(sentences=sentences, bert_data=bert_data)
            # for srldep_str in srldep_list:
            #
            #     srldep = {}
            #     srldep_dict = json.loads(srldep_str)
            #     for pred_id, argus in srldep_dict.items():
            #         srldep[int(pred_id)] = [(int(a[0]), a[2]) for a in argus]
            #
            #     srldep_pred.append(srldep)
            _, _, srldepdict = model(sentences=sentences, bert_data=bert_data)
            srldep_pred.extend(srldepdict)

        precision, recall, f1 = (
            srl_eval.compute_dependency_f1(self.ptb_dataset['brown_srldep_sent'], self.ptb_dataset['brown_srldep_dict'],
                                           srldep_pred,
                                           srl_conll_eval_path=False, use_gold=False))

        self.summary_dict['brown_srldep'] = FScore(recall, precision, f1)



