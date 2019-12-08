#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=0 python src_srl_syn/run_lm_finetuning.py test \
--model-path-base models/0702_2M_justbert_0_loss=100.pt