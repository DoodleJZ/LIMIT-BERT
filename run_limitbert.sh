#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python src_srl_syn/run_lm_finetuning.py train \
--save-model-path-base models/limit-bert \
--model-name limit-bert \
--model bert \
--use-electra \
--train-file bert_data/bert_train.txt \
--bert-model bert-large-uncased-whole-word-masking \
--max-seq-length 512 \
--train-batch-size 8 \
--num_train_epochs 1 \
--learning-rate 3e-5 \
--pre_step_tosave 5000 \
--pre_step 0 \
--pre_wiki_line 0 \
--p-ptb 0.0 \
--p-constmask 1 \
--p-srlmask 1 \
--p-wordmask 1 \
--p-tokenmask 0