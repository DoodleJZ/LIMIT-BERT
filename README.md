# LIMIT-BERT

## Contents
1. [Requirements](#Requirements)
2. [Training](#Training)

## Requirements

* Python 3.6 or higher.
* Cython 0.25.2 or any compatible version.
* [PyTorch](http://pytorch.org/) 1.0.0+. 
* [EVALB](http://nlp.cs.nyu.edu/evalb/). Before starting, run `make` inside the `EVALB/` directory to compile an `evalb` executable. This will be called from Python for evaluation. 
* [pytorch-transformers](https://github.com/huggingface/pytorch-transformers) PyTorch 1.0.0+ or any compatible version.

#### Pre-trained Models (PyTorch)
The following pre-trained models are available for download:
* [`LIMIT-BERT`](https://drive.google.com/open?id=1fm0cK2A91iLG3lCpwowCCQSALnWS2X4i): 
PyTorch版本,和BERT-wwm设置一致，Bert的PyTorch版直接加载

## Training

To train LIMIT-BERT, simply run:
```
sh run_limitbert.sh
```
### Evaluation Instructions

To test after setting model path:
```
sh test_bert.sh
```