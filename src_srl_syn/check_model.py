

import argparse
import itertools
import os.path
import time
import uuid
import torch
import numpy as np
import math
import json
from Evaluator import evaluate
from Evaluator import dep_eval
from Evaluator import srl_eval
from Evaluator import pos_eval

def format_elapsed(start_time):
    elapsed_time = int(time.time() - start_time)
    minutes, seconds = divmod(elapsed_time, 60)
    hours, minutes = divmod(minutes, 60)
    days, hours = divmod(hours, 24)
    elapsed_string = "{}h{:02}m{:02}s".format(hours, minutes, seconds)
    if days > 0:
        elapsed_string = "{}d{}".format(days, elapsed_string)
    return elapsed_string

class Check_model(object):
    def __init__(self, hparams, args, ptb_dataset):
        self.hparams = hparams
        self.ptb_dataset = ptb_dataset

        self.model_path_base = args.model_path_base
        self.log_path = args.log_path
        self.best_dev_score = -np.inf
        self.best_model_path = None
        self.model_path = None
        self.best_epoch = None
        self.eval_batch_size = args.eval_batch_size
        self.evalb_dir = args.evalb_dir

    def make_check(self, model, optimizer, epoch_num):

        print("Start dev eval:")
        summary_dict = {}

        dev_start_time = time.time()

        summary_dict["synconst dev F1"] = evaluate.FScore(0, 0, 0)
        summary_dict["syndep dev uas"] = 0
        summary_dict["syndep dev las"] = 0
        summary_dict["pos dev"] = 0
        summary_dict["synconst test F1"] = evaluate.FScore(0, 0, 0)
        summary_dict["syndep test uas"] = 0
        summary_dict["syndep test las"] = 0
        summary_dict["pos test"] = 0
        summary_dict["srlspan dev F1" ]= 0
        summary_dict["srldep dev F1"] = 0
        summary_dict["srlspan test F1"] = 0
        summary_dict["srlspan brown F1"] = 0
        summary_dict["srldep test F1"] = 0
        summary_dict["srldep brown F1"] = 0

        model.eval()

        syntree_pred = []
        srlspan_pred = []
        srldep_pred = []
        pos_pred = []
        if self.hparams.joint_syn:
            for start_index in range(0, len(self.ptb_dataset['dev_synconst_tree']), self.eval_batch_size):
                subbatch_trees = self.ptb_dataset['dev_synconst_tree'][start_index:start_index +self.eval_batch_size]
                subbatch_sentences = [[(leaf.tag, leaf.word) for leaf in tree.leaves()] for tree in subbatch_trees]

                syntree, _, _= model.parse_batch(subbatch_sentences)

                syntree_pred.extend(syntree)

            # const parsing:

            summary_dict["synconst dev F1"] = evaluate.evalb(self.evalb_dir, self.ptb_dataset['dev_synconst_tree'], syntree_pred)

            # dep parsing:

            dev_pred_head = [[leaf.father for leaf in tree.leaves()] for tree in syntree_pred]
            dev_pred_type = [[leaf.type for leaf in tree.leaves()] for tree in syntree_pred]
            assert len(dev_pred_head) == len(dev_pred_type)
            assert len(dev_pred_type) == len(self.ptb_dataset['dev_syndep_type'])
            summary_dict["syndep dev uas"], summary_dict["syndep dev las"] = dep_eval.eval(len(dev_pred_head), self.ptb_dataset['dev_syndep_sent'], self.ptb_dataset['dev_syndep_pos'],
                                             dev_pred_head, dev_pred_type, self.ptb_dataset['dev_syndep_head'], self.ptb_dataset['dev_syndep_type'],
                                             punct_set=self.hparams.punct_set, symbolic_root=False)
        # for srl different dev set
        if self.hparams.joint_srl or self.hparams.joint_pos:
            for start_index in range(0, len(self.ptb_dataset['dev_srlspan_sent']), self.eval_batch_size):
                subbatch_words = self.ptb_dataset['dev_srlspan_sent'][start_index:start_index + self.eval_batch_size]
                subbatch_pos = self.ptb_dataset['dev_srlspan_pos'][start_index:start_index + self.eval_batch_size]
                subbatch_sentences = [[(tag, word) for j, (tag, word) in enumerate(zip(tags, words))] for i, (tags, words)
                                      in enumerate(zip(subbatch_pos, subbatch_words))]

                srlspan_tree, srlspan_dict, _ = \
                    model(subbatch_sentences, gold_verbs=self.ptb_dataset['dev_srlspan_verb'][start_index:start_index + self.eval_batch_size])

                srlspan_pred.extend(srlspan_dict)
                pos_pred.extend([leaf.goldtag for leaf in srlspan_tree.leaves()])

            for start_index in range(0, len(self.ptb_dataset['dev_srldep_sent']), self.eval_batch_size):
                subbatch_words = self.ptb_dataset['dev_srldep_sent'][start_index:start_index + self.eval_batch_size]
                subbatch_pos = self.ptb_dataset['dev_srldep_pos'][start_index:start_index + self.eval_batch_size]
                subbatch_sentences = [[(tag, word) for j, (tag, word) in enumerate(zip(tags, words))] for i, (tags, words)
                                      in enumerate(zip(subbatch_pos, subbatch_words))]

                _, srldep_dict, _ = \
                    model(subbatch_sentences, gold_verbs=self.ptb_dataset['dev_srldep_verb'][
                                                         start_index:start_index + self.eval_batch_size])

                srldep_pred.extend(srldep_dict)

            if self.hparams.joint_srl:
                # srl span:
                # predicate F1
                # pid_precision, pred_recall, pid_f1, _, _, _, _ = srl_eval.compute_span_f1(
                #     srlspan_dev_verb, dev_pred_verb, "Predicate ID")
                print("===============================================")
                print("srl span dev eval:")
                precision, recall, f1, ul_prec, ul_recall, ul_f1 = (
                    srl_eval.compute_srl_f1(self.ptb_dataset['dev_srlspan_sent'], self.ptb_dataset['dev_srlspan_dict'], srlspan_pred, srl_conll_eval_path=False))
                summary_dict["srlspan dev F1"] = f1
                summary_dict["srlspan dev precision"] = precision
                summary_dict["srlspan dev recall"] = precision
                print("===============================================")
                print("srl dep dev eval:")
                precision, recall, f1 = (
                    srl_eval.compute_dependency_f1(self.ptb_dataset['dev_srldep_sent'], self.ptb_dataset['dev_srldep_dict'], srldep_pred,
                                                   srl_conll_eval_path=False, use_gold=self.hparams.use_gold_predicate))
                summary_dict["srldep dev F1"] = f1
                summary_dict["srldep dev precision"] = precision
                summary_dict["srldep dev recall"] = precision
                print("===============================================")

            if self.hparams.joint_pos:
                summary_dict["pos dev"] = pos_eval.eval(self.ptb_dataset['dev_srlspan_goldpos'], pos_pred)

        print(
            "dev-elapsed {} ".format(
                format_elapsed(dev_start_time),
            )
        )

        print(
            '============================================================================================================================')

        print("Start test eval:")
        test_start_time = time.time()

        syntree_pred = []
        srlspan_pred = []
        srldep_pred = []
        pos_pred = []
        test_fscore = evaluate.FScore(0, 0, 0)
        test_uas = 0
        test_las = 0
        for start_index in range(0, len(self.ptb_dataset['test_synconst_tree']), self.eval_batch_size):
            subbatch_trees = self.ptb_dataset['test_synconst_tree'][start_index:start_index + self.eval_batch_size]

            subbatch_sentences = [[(leaf.tag, leaf.word) for leaf in tree.leaves()] for tree in subbatch_trees]

            syntree, srlspan_dict, _ = \
                model(subbatch_sentences, gold_verbs=self.ptb_dataset['test_srlspan_verb'][
                                                     start_index:start_index + self.eval_batch_size])

            syntree_pred.extend(syntree)
            srlspan_pred.extend(srlspan_dict)
            pos_pred.extend([leaf.goldtag for leaf in syntree.leaves()])

        if self.hparams.joint_srl:
            for start_index in range(0, len(self.ptb_dataset['test_srlspan_sent']), self.eval_batch_size):

                subbatch_words_srldep = self.ptb_dataset['test_srlspan_sent'][start_index:start_index + self.eval_batch_size]
                subbatch_pos_srldep = self.ptb_dataset['test_srlspan_pos'][start_index:start_index + self.eval_batch_size]
                subbatch_sentences = [[(tag, word) for j, (tag, word) in enumerate(zip(tags, words))] for
                                             i, (tags, words)
                                             in enumerate(zip(subbatch_pos_srldep, subbatch_words_srldep))]

                _, _, srldep_dict = \
                    model(subbatch_sentences, gold_verbs=self.ptb_dataset['test_srldep_verb'][
                                                         start_index:start_index + self.eval_batch_size])

                srldep_pred.extend(srldep_dict)

            # const parsing:
        if self.hparams.joint_syn:
            summary_dict["synconst test F1"] = evaluate.evalb(self.evalb_dir, self.ptb_dataset['test_synconst_tree'], syntree_pred)

            # dep parsing:

            test_pred_head = [[leaf.father for leaf in tree.leaves()] for tree in syntree_pred]
            test_pred_type = [[leaf.type for leaf in tree.leaves()] for tree in syntree_pred]

            assert len(test_pred_head) == len(test_pred_type)
            assert len(test_pred_type) == len(self.ptb_dataset['test_syndep_type'])
            summary_dict["syndep test uas"], summary_dict["syndep test las"] = dep_eval.eval(len(test_pred_head), self.ptb_dataset['test_syndep_sent'], self.ptb_dataset['test_syndep_pos'], test_pred_head,
                                               test_pred_type, self.ptb_dataset['test_syndep_head'], self.ptb_dataset['test_syndep_type'],
                                               punct_set=self.hparams.punct_set, symbolic_root=False)

        if self.hparams.joint_pos:
            summary_dict["pos test"] = pos_eval.eval(self.ptb_dataset['test_srlspan_goldpos'], pos_pred)

        # srl span:
        if self.hparams.joint_srl:
            # predicate F1
            # pid_precision, pred_recall, pid_f1, _, _, _, _ = srl_eval.compute_span_f1(
            #     srlspan_test_verb, test_pred_verb, "Predicate ID")

            print("===============================================")
            print("wsj srl span test eval:")
            precision, recall, f1, ul_prec, ul_recall, ul_f1 = (
                srl_eval.compute_srl_f1(self.ptb_dataset['test_srlspan_sent'], self.ptb_dataset['test_srlspan_dict'], srlspan_pred, srl_conll_eval_path=False))
            summary_dict["srlspan test F1"] = f1
            summary_dict["srlspan test precision"] = precision
            summary_dict["srlspan test recall"] = precision
            print("===============================================")
            print("wsj srl dep test eval:")
            precision, recall, f1 = (
                srl_eval.compute_dependency_f1(self.ptb_dataset['test_srldep_sent'], self.ptb_dataset['test_srldep_dict'], srldep_pred,
                                               srl_conll_eval_path=False, use_gold=self.hparams.use_gold_predicate))
            summary_dict["srldep test F1"] = f1
            summary_dict["srldep test precision"] = precision
            summary_dict["srldep test recall"] = precision
            print("===============================================")

            print(
                '============================================================================================================================')

            syntree_pred = []
            srlspan_pred = []
            srldep_pred = []
            for start_index in range(0, len(self.ptb_dataset['brown_srlspan_sent']), self.eval_batch_size):
                subbatch_words = self.ptb_dataset['brown_srlspan_sent'][start_index:start_index + self.eval_batch_size]
                subbatch_pos = self.ptb_dataset['brown_srlspan_pos'][start_index:start_index + self.eval_batch_size]
                subbatch_sentences = [[(tag, word) for j, (tag, word) in enumerate(zip(tags, words))] for i, (tags, words)
                                      in enumerate(zip(subbatch_pos, subbatch_words))]

                syntree, srlspan_dict, _ = \
                    model(subbatch_sentences, gold_verbs=self.ptb_dataset['brown_srlspan_verb'][
                                                         start_index:start_index + self.eval_batch_size])
                syntree_pred.extend(syntree)
                srlspan_pred.extend(srlspan_dict)

            for start_index in range(0, len(self.ptb_dataset['brown_srldep_sent']), self.eval_batch_size):

                subbatch_words_srldep = self.ptb_dataset['brown_srldep_sent'][start_index:start_index + self.eval_batch_size]
                subbatch_pos_srldep = self.ptb_dataset['brown_srldep_sent'][start_index:start_index + self.eval_batch_size]
                subbatch_sentences = [[(tag, word) for j, (tag, word) in enumerate(zip(tags, words))] for
                                             i, (tags, words)
                                             in enumerate(zip(subbatch_pos_srldep, subbatch_words_srldep))]

                _, _, srldep_dict = \
                    model(subbatch_sentences, gold_verbs=self.ptb_dataset['brown_srldep_verb'][
                                                         start_index:start_index + self.eval_batch_size])

                srldep_pred.extend(srldep_dict)

            # predicate F1
            # pid_precision, pred_recall, pid_f1, _, _, _, _ = srl_eval.compute_span_f1(
            #     srlspan_test_verb, test_pred_verb, "Predicate ID")

            print("===============================================")
            print("brown srl span test eval:")
            precision, recall, f1, ul_prec, ul_recall, ul_f1 = (
                srl_eval.compute_srl_f1(self.ptb_dataset['brown_srlspan_sent'], self.ptb_dataset['brown_srlspan_dict'], srlspan_pred, srl_conll_eval_path=False))
            summary_dict["srlspan brown F1"] = f1
            summary_dict["srlspan brown precision"] = precision
            summary_dict["srlspan brown recall"] = precision
            print("===============================================")
            print("brown srl dep test eval:")
            precision, recall, f1 = (
                srl_eval.compute_dependency_f1(self.ptb_dataset['brown_srldep_sent'], self.ptb_dataset['brown_srldep_dict'], srldep_pred, srl_conll_eval_path=False,
                                               use_gold=self.hparams.use_gold_predicate))
            summary_dict["srldep brown F1"] = f1
            summary_dict["srldep brown precision"] = precision
            summary_dict["srldep brown recall"] = precision
            print("===============================================")

        print(
            "test-elapsed {} ".format(
                format_elapsed(test_start_time)
            )
        )

        print(
            '============================================================================================================================')

        if summary_dict['synconst dev F1'].fscore + summary_dict['syndep dev las'] + summary_dict["srlspan dev F1"] + summary_dict[
            "srldep dev F1"] + summary_dict['pos dev'] > self.best_dev_score:
            if self.best_model_path is not None:
                extensions = [".pt"]
                for ext in extensions:
                    path = self.best_model_path + ext
                    if os.path.exists(path):
                        print("Removing previous model file {}...".format(path))
                        os.remove(path)

            self.best_dev_score = summary_dict['synconst dev F1'].fscore + summary_dict['syndep dev las'] + summary_dict["srlspan dev F1"] + summary_dict[
            "srldep dev F1"] + summary_dict['pos dev']
            best_model_path = "{}_best_dev={:.2f}_devuas={:.2f}_devlas={:.2f}_devsrlspan={:.2f}_devsrldep={:.2f}".format(
                self.model_path_base, summary_dict['synconst dev F1'], summary_dict['syndep dev uas'], summary_dict['syndep dev las'],
                summary_dict["srlspan dev F1"], summary_dict["srldep dev F1"])
            print("Saving new best model to {}...".format(best_model_path))
            torch.save({
                'spec': model.spec,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
            }, best_model_path + ".pt")

        log_data = "{} epoch, dev-fscore {:},test-fscore {:}, dev-uas {:.2f}, dev-las {:.2f}，" \
                   "test-uas {:.2f}, test-las {:.2f}, dev-srlspan {:.2f}, test-wsj-srlspan {:.2f}, test-brown-srlspan {:.2f}," \
                   " dev-srldep {:.2f},  test-wsj-srldep {:.2f}, test-brown-srldep {:.2f}, dev-pos {:.2f}, test-pos {:.2f}," \
                   "dev_score {:.2f}, best_dev_score {:.2f}" \
            .format(epoch_num, summary_dict["synconst dev F1"], summary_dict["synconst test F1"],
                    summary_dict["syndep dev uas"], summary_dict["syndep dev las"],
                    summary_dict["syndep test uas"], summary_dict["syndep test las"],
                    summary_dict["srlspan dev F1"], summary_dict["srlspan test F1"], summary_dict["srlspan brown F1"],
                    summary_dict["srldep dev F1"], summary_dict["srldep test F1"], summary_dict["srldep brown F1"],
                    summary_dict["pos dev"], summary_dict["pos test"],
                    summary_dict['synconst dev F1'].fscore + summary_dict['syndep dev las'] + summary_dict["srlspan dev F1"] +
                    summary_dict["srldep dev F1"] + summary_dict['pos dev'],
                    self.best_dev_score)

        if not os.path.exists(self.log_path):
            flog = open(self.log_path, 'w')
        flog = open(self.log_path, 'r+')
        content = flog.read()
        flog.seek(0, 0)
        flog.write(log_data + '\n' + content)