# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HugginFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""BERT finetuning runner."""

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
from Eval import EvalManyTask
import makehp
import json
import Zmodel
import trees
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


def warmup_linear(x, warmup=0.002):
    if x < warmup:
        return x/warmup
    return 1.0 - x

def torch_load(load_path):
    if Zmodel.use_cuda:
        return torch.load(load_path)
    else:
        return torch.load(load_path, map_location=lambda storage, location: storage)

def make_hparams():
    return makehp.HParams(
        max_len_train=0, # no length limit
        max_len_dev=0, # no length limit

        max_seq_length=256,

        learning_rate=3e-5,
        # learning_rate_warmup_steps=160,
        # clip_grad_norm=0., #no clipping
        # step_decay=True, # note that disabling step decay is not implemented
        # step_decay_factor=0.5,
        # step_decay_patience=5,

        #Probability setting
        p_ptb = 0.5,
        p_constmask = 5,
        p_srlmask  = 1,
        p_wordmask = 1,
        p_tokenmask = 0,

        #Joint setting
        partitioned=True,
        use_only_bert = False,
        # use_cat=False,
        # use_lstm = False,
        joint_syn = True,
        joint_srl = True,
        joint_pos = True,

        #SRL setting
        const_lada = 0.5,
        labmda_verb = 0.6,
        labmda_span = 0.6,
        max_num_span = 300,
        max_num_verb = 30,
        use_srl_jointdecode = True,

        #Task layer setting
        num_layers=2,
        d_model=1024,
        num_heads=8,
        d_kv=64,
        d_ff=2048,
        d_label_hidden=250,
        d_biaffine = 1024,
        d_score_hidden = 256,
        d_span = 512,

        attention_dropout=0.2,
        embedding_dropout=0.2,
        relu_dropout=0.2,
        residual_dropout=0.2,
        morpho_emb_dropout=0.2,
        timing_dropout=0.0,

        model_name = "multitask-bert",
        punctuation='.' '``' "''" ':' ',',

        model = "bert",
        use_sparse = False,
        use_electra = False,
        use_alltoken = False,
        bert_model="",#""bert-base-uncased",
        bert_do_lower_case=True,
        bert_transliterate="",
        )

def format_elapsed(start_time):
    elapsed_time = int(time.time() - start_time)
    minutes, seconds = divmod(elapsed_time, 60)
    hours, minutes = divmod(minutes, 60)
    days, hours = divmod(hours, 24)
    elapsed_string = "{}h{:02}m{:02}s".format(hours, minutes, seconds)
    if days > 0:
        elapsed_string = "{}d{}".format(days, elapsed_string)
    return elapsed_string

def run_train(args, hparams):
    if args.seed is not None:
        print("Setting numpy random seed to {}...".format(args.seed))
        np.random.seed(args.seed)

    seed_from_numpy = np.random.randint(2147483648)
    print("Manual seed for pytorch:", seed_from_numpy)
    torch.manual_seed(seed_from_numpy)

    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        n_gpu = torch.cuda.device_count()
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        n_gpu = 1
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.distributed.init_process_group(backend='nccl')
    logger.info("device: {} n_gpu: {}, distributed training: {}, 16-bits training: {}".format(
        device, n_gpu, bool(args.local_rank != -1), args.fp16))

    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
                            args.gradient_accumulation_steps))

    args.train_batch_size = int(args.train_batch_size / args.gradient_accumulation_steps)

    if n_gpu > 0:
        torch.cuda.manual_seed_all(seed_from_numpy)

    # if os.path.exists(args.output_dir) and os.listdir(args.output_dir):
    #     raise ValueError("Output directory ({}) already exists and is not empty.".format(args.output_dir))
    # os.makedirs(args.output_dir, exist_ok=True)

    print("Initializing model...")
    load_path = args.load_path
    if load_path is not None :
        print(f"Loading parameters from {load_path}")
        info = torch_load(load_path)
        model = Zmodel.Jointmodel.from_spec(info['spec'], info['state_dict'])
        hparams = model.hparams
        Ptb_dataset = PTBDataset(hparams)
        Ptb_dataset.process_PTB(args)
    else:
        hparams.set_from_args(args)
        Ptb_dataset = PTBDataset(hparams)
        Ptb_dataset.process_PTB(args)
        model = Zmodel.Jointmodel(
            Ptb_dataset.tag_vocab,
            Ptb_dataset.word_vocab,
            Ptb_dataset.label_vocab,
            Ptb_dataset.char_vocab,
            Ptb_dataset.type_vocab,
            Ptb_dataset.srl_vocab,
            hparams,
        )
    print("Hyperparameters:")
    hparams.print()
    # tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=args.do_lower_case)

    #train_examples = None
    # num_train_steps = None
    print("Loading Train Dataset", args.train_file)
    Ptb_dataset.rand_dataset()
    # print(model.tokenizer.tokenize("Federal Paper Board sells paper and wood products ."))
    #max_seq_length = model.bert_max_len
    train_dataset = BERTDataset(args.pre_wiki_line, hparams, Ptb_dataset, args.train_file, model.tokenizer, seq_len= args.max_seq_length,
                                corpus_lines=None, on_memory=args.on_memory)
    task_list = ['dev_synconst', 'dev_srlspan', 'dev_srldep', 'test_synconst', 'test_srlspan', 'test_srldep', 'brown_srlspan', 'brown_srldep']
    evaluator = EvalManyTask(device = 1, hparams = hparams, ptb_dataset = Ptb_dataset, task_list = task_list, bert_tokenizer = model.tokenizer, seq_len = args.eval_seq_length, eval_batch_size = args.eval_batch_size,
                             evalb_dir = args.evalb_dir, model_path_base = args.save_model_path_base, log_path = "{}_log".format("models_log/" + hparams.model_name))

    num_train_steps = int(
        len(train_dataset) / args.train_batch_size / args.gradient_accumulation_steps * args.num_train_epochs)

    # Prepare model
    # model = BertForPreTraining.from_pretrained(args.bert_model)
    if args.fp16:
        model.half()
    model.to(device)
    if args.local_rank != -1:
        try:
            from apex.parallel import DistributedDataParallel as DDP
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")
        model = DDP(model)
    elif n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Prepare optimizer
    param_optimizer = list(model.named_parameters())

    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
    if args.fp16:
        try:
            from apex.optimizers import FP16_Optimizer
            from apex.optimizers import FusedAdam
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")

        optimizer = FusedAdam(optimizer_grouped_parameters,
                              lr=args.learning_rate,
                              bias_correction=False,
                              max_grad_norm=1.0)
        if args.loss_scale == 0:
            optimizer = FP16_Optimizer(optimizer, dynamic_loss_scale=True)
        else:
            optimizer = FP16_Optimizer(optimizer, static_loss_scale=args.loss_scale)

    else:
        optimizer = BertAdam(optimizer_grouped_parameters,
                             lr=args.learning_rate,
                             warmup=args.warmup_proportion,
                             t_total=num_train_steps)
    if load_path is not None:
        optimizer.load_state_dict(info['optimizer'])


    global_step = args.pre_step
    pre_step = args.pre_step
    # wiki_line = 0
    # while train_dataset.wiki_id < wiki_line:
    #     train_dataset.file.__next__().strip()
    #     train_dataset.wiki_id+=1

    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Batch size = %d", args.train_batch_size)
    logger.info("  Num steps = %d", num_train_steps)

    if args.local_rank == -1:
        train_sampler = RandomSampler(train_dataset)
    else:
        #TODO: check if this works with current data generator from disk that relies on file.__next__
        # (it doesn't return item back by index)
        train_sampler = DistributedSampler(train_dataset)

    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size)
    hparams.model_name = args.model_name
    print("This is ", hparams.model_name)
    start_time = time.time()

    def save_args(hparams):
        arg_path = "{}_log".format("models_log/" + hparams.model_name) + '.arg.json'
        kwargs = hparams.to_dict()
        json.dump({'kwargs': kwargs}, open(arg_path, 'w'), indent=4)

    save_args(hparams)
    # test_save_path = args.save_model_path_base + "_fortest"
    # torch.save({
    #     'spec': model_to_save.spec,
    #     'state_dict': model_to_save.state_dict(),
    #     'optimizer': optimizer.state_dict(),
    # }, test_save_path + ".pt")
    # evaluator.test_model_path = test_save_path

    cur_ptb_epoch = 0
    for epoch in trange(int(args.num_train_epochs), desc="Epoch"):

        tr_loss = 0
        nb_tr_examples, nb_tr_steps = 0, 0

        #save_model_path, is_save = evaluator.eval_multitask(start_time, cur_ptb_epoch)

        epoch_start_time = time.time()
        for step, batch in enumerate(train_dataloader):
            model.train()
            input_ids, origin_ids, input_mask, word_start_mask, word_end_mask, segment_ids, perm_mask, target_mapping, lm_label_ids, lm_label_mask, is_next, \
            synconst_list, syndep_head_list, syndep_type_list, srlspan_str_list, srldep_str_list, is_ptb = batch
            # synconst_list, syndep_head_list, syndep_type_list , srlspan_str_list, srldep_str_list = gold_list
            dis_idx = [i for i in range(len(input_ids))]
            dis_idx = torch.tensor(dis_idx)
            batch = dis_idx, input_ids, origin_ids, input_mask, word_start_mask, word_end_mask, segment_ids, perm_mask, target_mapping, lm_label_ids, lm_label_mask, is_next
            bert_data = tuple(t.to(device) for t in batch)
            sentences = []
            gold_syntree = []
            gold_srlspans = []
            gold_srldeps = []

            # for data_dict1 in dict1:
            for synconst, syndep_head_str, syndep_type_str, srlspan_str, srldep_str in zip(synconst_list, syndep_head_list, syndep_type_list, srlspan_str_list, srldep_str_list):

                syndep_head = json.loads(syndep_head_str)
                syndep_type = json.loads(syndep_type_str)
                syntree = trees.load_trees(synconst, [[int(head) for head in syndep_head]], [syndep_type], strip_top = False)[0]
                sentences.append([(leaf.tag, leaf.word) for leaf in syntree.leaves()])

                gold_syntree.append(syntree.convert())

                srlspan = {}
                srlspan_dict = json.loads(srlspan_str)
                for pred_id, argus in srlspan_dict.items():
                    srlspan[int(pred_id)] = [(int(a[0]), int(a[1]), a[2]) for a in argus]

                srldep_dict = json.loads(srldep_str)
                srldep = {}
                if str(-1) in srldep_dict:
                    srldep = None
                else:
                    for pred_id, argus in srldep_dict.items():
                        srldep[int(pred_id)] = [(int(a[0]), a[1]) for a in argus]

                gold_srlspans.append(srlspan)
                gold_srldeps.append(srldep)

            if global_step < pre_step:
                if global_step % 1000 == 0:
                    print("global_step:", global_step)
                    print("pre_step:", pre_step)
                    print("Wiki line:", train_dataset.wiki_line)
                    print("total-elapsed {} ".format(format_elapsed(start_time)))
                global_step += 1
                cur_ptb_epoch = train_dataset.ptb_epoch
                continue

            bert_loss, task_loss = model(sentences = sentences, gold_trees = gold_syntree, gold_srlspans = gold_srlspans, gold_srldeps = gold_srldeps, bert_data = bert_data)
            if n_gpu > 1:
                bert_loss = bert_loss.sum()
                task_loss = task_loss.sum()
            loss = bert_loss + task_loss #* 0.1
            loss = loss / len(synconst_list)
            bert_loss = bert_loss / len(synconst_list)
            task_loss = task_loss / len(synconst_list)

            tatal_loss = float(loss.data.cpu().numpy())

            if bert_loss > 0:
                bert_loss = float(bert_loss.data.cpu().numpy())
            if task_loss > 0:
                task_loss = float(task_loss.data.cpu().numpy())

            # grad_norm = torch.nn.utils.clip_grad_norm_(clippable_parameters, grad_clip_threshold)

            lr_this_step = args.learning_rate * warmup_linear(global_step / num_train_steps, args.warmup_proportion)
            print(
                "epoch {:,} "
                "ptb-epoch {:,} "
                "batch {:,}/{:,} "
                "processed {:,} "
                "PTB line {:,} "
                "Wiki line {:,} "
                "total-loss {:.4f} "
                "bert-loss {:.4f} "
                "task-loss {:.4f} "
                "lr_this_step {:.12f} "
                "epoch-elapsed {} "
                "total-elapsed {}".format(
                    epoch,
                    cur_ptb_epoch,
                    global_step,
                    int(np.ceil(len(train_dataset) / args.train_batch_size)),
                    (global_step + 1) * args.train_batch_size,
                    train_dataset.ptb_cur_line,
                    train_dataset.wiki_line,
                    tatal_loss,
                    bert_loss,
                    task_loss,
                    lr_this_step,
                    format_elapsed(epoch_start_time),
                    format_elapsed(start_time),
                )
            )

            # if n_gpu > 1:
            #     loss = loss.mean() # mean() to average on multi-gpu.
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps
            if args.fp16:
                optimizer.backward(loss)
            else:
                loss.backward()
            tr_loss += loss.item()
            nb_tr_examples += input_ids.size(0)
            nb_tr_steps += 1
            if (step + 1) % args.gradient_accumulation_steps == 0:
                # modify learning rate with special warm up BERT uses
                lr_this_step = args.learning_rate * warmup_linear(global_step/num_train_steps, args.warmup_proportion)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr_this_step
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1

                #if train_dataset.ptb_epoch > cur_ptb_epoch:
                if global_step % args.pre_step_tosave == 0:
                    cur_ptb_epoch = train_dataset.ptb_epoch

                    save_path = "{}_gstep{}_wiki{}_loss={:.4f}.pt".\
                        format(args.save_model_path_base, global_step, train_dataset.wiki_line, tatal_loss)
                    model_to_save = model.module if hasattr(model, 'module') else model
                    torch.save({
                        'spec': model_to_save.spec,
                        'state_dict': model_to_save.state_dict(),
                        'optimizer': optimizer.state_dict(),
                    }, save_path)

                    # evaluator.test_model_path = test_save_path
                    #
                    # save_model_path, is_save = evaluator.eval_multitask(start_time, cur_ptb_epoch)
                    # if is_save:
                    #     print("Saving new best model to {}...".format(save_model_path))
                    #     torch.save({
                    #         'spec': model_to_save.spec,
                    #         'state_dict': model_to_save.state_dict(),
                    #         'optimizer': optimizer.state_dict(),
                    #     }, save_model_path + ".pt")


    # Save a trained model
    logger.info("** ** * Saving fine - tuned model ** ** * ")
    torch.save({
        'spec': model_to_save.spec,
        'state_dict': model_to_save.state_dict(),
        'optimizer': optimizer.state_dict(),
    }, args.save_model_path_base + ".pt")

def run_test(args):


    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    n_gpu = torch.cuda.device_count()

    print("Loading model from {}...".format(args.model_path_base))
    assert args.model_path_base.endswith(".pt"), "Only pytorch savefiles supported"

    info = torch_load(args.model_path_base)
    assert 'hparams' in info['spec'], "Older savefiles not supported"
    model = Zmodel.Jointmodel.from_spec(info['spec'], info['state_dict'])
    model.to(device)
    model.hparams.bert_model = "bert-large-uncased-whole-word-masking"
    Ptb_dataset = PTBDataset(model.hparams)
    Ptb_dataset.process_PTB(args)

    task_list = ['dev_synconst', 'dev_srlspan', 'dev_srldep', 'test_synconst', 'test_srlspan', 'test_srldep',
                 'brown_srlspan', 'brown_srldep']
    length = 0
    for i in range(10):
        length += 10
        task_list.append('dev_synconst'+str(length))
        task_list.append('dev_srlspan' + str(length))
        task_list.append('dev_srldep' + str(length))
    evaluator = EvalManyTask(device=device, hparams=model.hparams, ptb_dataset=Ptb_dataset, task_list=task_list,
                             bert_tokenizer=model.tokenizer, seq_len=args.eval_seq_length,
                             eval_batch_size=args.eval_batch_size,
                             evalb_dir=args.evalb_dir, model_path_base=args.model_path_base,
                             log_path="{}_log".format("models_log/" + model.hparams.model_name))

    start_time = time.time()

    save_model_path, is_save = evaluator.eval_multitask(model, start_time, 0)




def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()

    hparams = make_hparams()
    subparser = subparsers.add_parser("train")
    subparser.set_defaults(callback=lambda args: run_train(args, hparams))
    hparams.populate_arguments(subparser)
    subparser.add_argument('--load_path', type=str, default=None)
    subparser.add_argument('--seed',
                           type=int,
                           default=42,
                           help="random seed for initialization")
    subparser.add_argument("--bert-model", default=None, type=str, required=True,
                           help="Bert pre-trained model selected in the list: bert-base-uncased, "
                                "bert-large-uncased, bert-base-cased, bert-base-multilingual, bert-base-chinese.")
    # subparser.add_argument("--max-seq-length",
    #                        default=128,
    #                        type=int,
    #                        help="The maximum total input sequence length after WordPiece tokenization. \n"
    #                             "Sequences longer than this will be truncated, and sequences shorter \n"
    #                             "than this will be padded.")
    # subparser.add_argument("--learning-rate",
    #                        default=3e-5,
    #                        type=float,
    #                        help="The initial learning rate for Adam.")
    subparser.add_argument("--warmup_proportion",
                           default=0.1,
                           type=float,
                           help="Proportion of training to perform linear learning rate warmup for. "
                                "E.g., 0.1 = 10%% of training.")
    subparser.add_argument("--num_train_epochs",
                           default=3.0,
                           type=float,
                           help="Total number of training epochs to perform.")
    subparser.add_argument("--pre_step_tosave",
                           default=10000,
                           type=int,
                           help="Save the models pre step.")
    subparser.add_argument("--pre_wiki_line",
                           default=0,
                           type=int,
                           help="If load pre-trained model, begin with last wiki line.")
    subparser.add_argument("--pre_step",
                           default=0,
                           type=int,
                           help="If load pre-trained model, begin with last step.")
    subparser.add_argument("--no_cuda",
                           action='store_true',
                           help="Whether not to use CUDA when available")
    subparser.add_argument("--on_memory",
                           action='store_true',
                           help="Whether to load train samples into memory or use disk")
    subparser.add_argument("--do_lower_case",
                           action='store_true',
                           help="Whether to lower case the input text. True for uncased models, False for cased models.")
    subparser.add_argument("--local_rank",
                           type=int,
                           default=-1,
                           help="local_rank for distributed training on gpus")
    subparser.add_argument('--gradient_accumulation_steps',
                           type=int,
                           default=1,
                           help="Number of updates steps to accumualte before performing a backward/update pass.")
    subparser.add_argument('--fp16',
                           action='store_true',
                           help="Whether to use 16-bit float precision instead of 32-bit")
    subparser.add_argument('--loss_scale',
                           type=float, default=0,
                           help="Loss scaling to improve fp16 numeric stability. Only used when fp16 set to True.\n"
                                "0 (default value): dynamic loss scaling.\n"
                                "Positive power of 2: static loss scaling value.\n")

    subparser.add_argument("--train-file",
                           default="bert_data/bert_train.txt",
                           type=str,
                           required=True,
                           help="The input train corpus.")
    subparser.add_argument("--train-batch-size",
                           default=32,
                           type=int,
                           help="Total batch size for training.")


    subparser.add_argument("--model-name", default="test")
    subparser.add_argument("--model", default="bert")
    subparser.add_argument("--evalb-dir", default="EVALB/")
    subparser.add_argument("--save-model-path-base", required=True)
    subparser.add_argument("--print-vocabs", action="store_true")

    #PTB setting:
    subparser.add_argument("--srldep-align-path", default="data/seldep_train_align_path.json")

    subparser.add_argument("--synconst-train-ptb-path", default="data/02-21.10way.clean")
    subparser.add_argument("--synconst-dev-ptb-path", default="data/22.auto.clean")
    subparser.add_argument("--syndep-train-ptb-path", default="data/ptb_train_3.3.0.sd")
    subparser.add_argument("--syndep-dev-ptb-path", default="data/ptb_dev_3.3.0.sd")
    subparser.add_argument("--srlspan-train-ptb-path", default="data/srl_span_train.txt")
    subparser.add_argument("--srlspan-dev-ptb-path", default="data/srl_span_dev.txt")
    subparser.add_argument("--srldep-train-ptb-path", default="data/srl_dep_train.txt")
    subparser.add_argument("--srldep-dev-ptb-path", default="data/srl_dep_dev.txt")

    subparser.add_argument("--eval_batch_size",
                           default=8,
                           type=int,
                           help="Total batch size for eval.")
    subparser.add_argument("--eval_seq_length",
                           default=350,
                           type=int)
    subparser.add_argument("--synconst-test-ptb-path", default="data/23.auto.clean")
    subparser.add_argument("--syndep-test-ptb-path", default="data/ptb_test_3.3.0.sd")
    subparser.add_argument("--srlspan-test-ptb-path", default="data/srl_span_testwsj.txt")
    subparser.add_argument("--srlspan-test-brown-path", default="data/srl_span_testbrown.txt")
    subparser.add_argument("--srldep-test-ptb-path", default="data/srl_dep_testwsj.txt")
    subparser.add_argument("--srldep-test-brown-path", default="data/srl_dep_testbrown.txt")

    subparser = subparsers.add_parser("test")
    subparser.set_defaults(callback=run_test)
    subparser.add_argument("--model-path-base", required=True)
    subparser.add_argument("--model-name", default="test")
    subparser.add_argument("--model", default="bert")
    subparser.add_argument("--evalb-dir", default="EVALB/")
    subparser.add_argument("--no_cuda",
                           action='store_true',
                           help="Whether not to use CUDA when available")
    subparser.add_argument("--print-vocabs", action="store_true")

    subparser.add_argument("--srldep-align-path", default="data/seldep_train_align_path.json")

    subparser.add_argument("--synconst-train-ptb-path", default="data/02-21.10way.clean")
    subparser.add_argument("--synconst-dev-ptb-path", default="data/22.auto.clean")
    subparser.add_argument("--syndep-train-ptb-path", default="data/ptb_train_3.3.0.sd")
    subparser.add_argument("--syndep-dev-ptb-path", default="data/ptb_dev_3.3.0.sd")
    subparser.add_argument("--srlspan-train-ptb-path", default="data/srl_span_train.txt")
    subparser.add_argument("--srlspan-dev-ptb-path", default="data/srl_span_dev.txt")
    subparser.add_argument("--srldep-train-ptb-path", default="data/srl_dep_train.txt")
    subparser.add_argument("--srldep-dev-ptb-path", default="data/srl_dep_dev.txt")

    subparser.add_argument("--eval_batch_size",
                           default=8,
                           type=int,
                           help="Total batch size for eval.")
    subparser.add_argument("--eval_seq_length",
                           default=256,
                           type=int)
    subparser.add_argument("--synconst-test-ptb-path", default="data/23.auto.clean")
    subparser.add_argument("--syndep-test-ptb-path", default="data/ptb_test_3.3.0.sd")
    subparser.add_argument("--srlspan-test-ptb-path", default="data/srl_span_testwsj.txt")
    subparser.add_argument("--srlspan-test-brown-path", default="data/srl_span_testbrown.txt")
    subparser.add_argument("--srldep-test-ptb-path", default="data/srl_dep_testwsj.txt")
    subparser.add_argument("--srldep-test-brown-path", default="data/srl_dep_testbrown.txt")
    #
    # subparser = subparsers.add_parser("parse")
    # subparser.set_defaults(callback=run_parse)
    # subparser.add_argument("--model-path-base", required=True)
    # subparser.add_argument("--embedding-path", default="data/glove.6B.100d.txt.gz")
    # subparser.add_argument("--dataset", default="ptb")
    # subparser.add_argument("--input-path", type=str, required=True)
    # subparser.add_argument("--output-path-synconst", type=str, default="-")
    # subparser.add_argument("--output-path-syndep", type=str, default="-")
    # subparser.add_argument("--output-path-synlabel", type=str, default="-")
    # subparser.add_argument("--output-path-hpsg", type=str, default="-")
    # subparser.add_argument("--output-path-srlspan", type=str, default="-")
    # subparser.add_argument("--output-path-srldep", type=str, default="-")
    # subparser.add_argument("--eval-batch-size", type=int, default=50)


    args = parser.parse_args()
    args.callback(args)

if __name__ == "__main__":
    main()