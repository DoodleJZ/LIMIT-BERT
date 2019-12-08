import functools

import numpy as np

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.init as init
from torch.utils.data import Dataset
import random
import logging
logger = logging.getLogger(__name__)

use_cuda = torch.cuda.is_available()
if use_cuda:
    torch_t = torch
    def from_numpy(ndarray):
        npy = torch.tensor(ndarray)
        return npy
    # torch_t = torch.cuda
    # def from_numpy(ndarray):
    #     return torch.from_numpy(ndarray).pin_memory().cuda(async=True)
else:
    print("Not using CUDA!")
    torch_t = torch
    from torch import from_numpy

import pyximport
pyximport.install(setup_args={"include_dirs": np.get_include()})
#import src_dep_const_test.chart_helper as chart_helper
from Decoder.uniform_decoder import Uniform_Decoder
import makehp
import utils
import json
import trees

START = "<START>"
STOP = "<STOP>"
UNK = "<UNK>"
ROOT = "<START>"
Sub_Head = "<H>"
No_Head = "<N>"

TAG_UNK = "UNK"

ROOT_TYPE = "<ROOT_TYPE>"

# Assumes that these control characters are not present in treebank text
CHAR_UNK = "\0"
CHAR_START_SENTENCE = "\1"
CHAR_START_WORD = "\2"
CHAR_STOP_WORD = "\3"
CHAR_STOP_SENTENCE = "\4"
CHAR_PAD = "\5"

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


class BatchIndices:
    """
    Batch indices container class (used to implement packed batches)
    """
    def __init__(self, batch_idxs_np):
        self.batch_idxs_np = batch_idxs_np
        self.batch_idxs_torch = from_numpy(batch_idxs_np)

        self.batch_size = int(1 + np.max(batch_idxs_np))

        batch_idxs_np_extra = np.concatenate([[-1], batch_idxs_np, [-1]])
        self.boundaries_np = np.nonzero(batch_idxs_np_extra[1:] != batch_idxs_np_extra[:-1])[0]
        self.seq_lens_np = self.boundaries_np[1:] - self.boundaries_np[:-1]
        assert len(self.seq_lens_np) == self.batch_size
        self.max_len = int(np.max(self.boundaries_np[1:] - self.boundaries_np[:-1]))

#
class FeatureDropoutFunction(torch.autograd.function.InplaceFunction):
    @classmethod
    def forward(cls, ctx, input, batch_idxs, p=0.5, train=False, inplace=False):
        if p < 0 or p > 1:
            raise ValueError("dropout probability has to be between 0 and 1, "
                             "but got {}".format(p))

        ctx.p = p
        ctx.train = train
        ctx.inplace = inplace

        if ctx.inplace:
            ctx.mark_dirty(input)
            output = input
        else:
            output = input.clone()

        if ctx.p > 0 and ctx.train:
            ctx.noise = input.new().resize_(batch_idxs.batch_size, input.size(1))
            if ctx.p == 1:
                ctx.noise.fill_(0)
            else:
                ctx.noise.bernoulli_(1 - ctx.p).div_(1 - ctx.p)
            ctx.noise = ctx.noise[batch_idxs.batch_idxs_torch, :]
            output.mul_(ctx.noise)

        return output

    @staticmethod
    def backward(ctx, grad_output):
        if ctx.p > 0 and ctx.train:
            return grad_output.mul(Variable(ctx.noise)), None, None, None, None
        else:
            return grad_output, None, None, None, None

#
class FeatureDropout(nn.Module):
    """
    Feature-level dropout: takes an input of size len x num_features and drops
    each feature with probabibility p. A feature is dropped across the full
    portion of the input that corresponds to a single batch element.
    """
    def __init__(self, p=0.5, inplace=False):
        super().__init__()
        if p < 0 or p > 1:
            raise ValueError("dropout probability has to be between 0 and 1, "
                             "but got {}".format(p))
        self.p = p
        self.inplace = inplace

    def forward(self, input, batch_idxs):
        return FeatureDropoutFunction.apply(input, batch_idxs, self.p, self.training, self.inplace)

#
class LayerNormalization(nn.Module):
    def __init__(self, d_hid, eps=1e-3, affine=True):
        super(LayerNormalization, self).__init__()

        self.eps = eps
        self.affine = affine
        if self.affine:
            self.a_2 = nn.Parameter(torch.ones(d_hid), requires_grad=True)
            self.b_2 = nn.Parameter(torch.zeros(d_hid), requires_grad=True)

    def forward(self, z):
        if z.size(-1) == 1:
            return z

        mu = torch.mean(z, keepdim=True, dim=-1)
        sigma = torch.std(z, keepdim=True, dim=-1)
        ln_out = (z - mu.expand_as(z)) / (sigma.expand_as(z) + self.eps)
        if self.affine:
            ln_out = ln_out * self.a_2.expand_as(ln_out) + self.b_2.expand_as(ln_out)

        return ln_out

#
class ScaledAttention(nn.Module):
    def __init__(self, hparams, attention_dropout=0.1):
        super(ScaledAttention, self).__init__()
        self.hparams = hparams
        self.temper = hparams.d_model ** 0.5
        self.dropout = nn.Dropout(attention_dropout)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, q, k, v, attn_mask=None):
        # q: [batch, slot, feat]
        # k: [batch, slot, feat]
        # v: [batch, slot, feat]

        attn = torch.bmm(q, k.transpose(1, 2)) / self.temper


        if attn_mask is not None:
            assert attn_mask.size() == attn.size(), \
                    'Attention mask shape {} mismatch ' \
                    'with Attention logit tensor shape ' \
                    '{}.'.format(attn_mask.size(), attn.size())

            attn.data.masked_fill_(attn_mask, -float('inf'))

        attn = self.softmax(attn.transpose(1, 2)).transpose(1, 2)
        attn = self.dropout(attn)
        output = torch.bmm(attn, v)

        return output, attn

#
class MultiHeadAttention(nn.Module):
    """
    Multi-head attention module
    """

    def __init__(self, hparams, n_head, d_model, d_k, d_v, residual_dropout=0.1, attention_dropout=0.1, d_positional=None):
        super(MultiHeadAttention, self).__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v
        self.hparams = hparams

        if d_positional is None:
            self.partitioned = False
        else:
            self.partitioned = True

        if self.partitioned:
            self.d_content = d_model - d_positional
            self.d_positional = d_positional

            self.w_qs1 = nn.Parameter(torch_t.Tensor(n_head, self.d_content, d_k // 2))
            self.w_ks1 = nn.Parameter(torch_t.Tensor(n_head, self.d_content, d_k // 2))
            self.w_vs1 = nn.Parameter(torch_t.Tensor(n_head, self.d_content, d_v // 2))

            self.w_qs2 = nn.Parameter(torch_t.Tensor(n_head, self.d_positional, d_k // 2))
            self.w_ks2 = nn.Parameter(torch_t.Tensor(n_head, self.d_positional, d_k // 2))
            self.w_vs2 = nn.Parameter(torch_t.Tensor(n_head, self.d_positional, d_v // 2))

            init.xavier_normal_(self.w_qs1)
            init.xavier_normal_(self.w_ks1)
            init.xavier_normal_(self.w_vs1)

            init.xavier_normal_(self.w_qs2)
            init.xavier_normal_(self.w_ks2)
            init.xavier_normal_(self.w_vs2)
        else:
            self.w_qs = nn.Parameter(torch_t.Tensor(n_head, d_model, d_k))
            self.w_ks = nn.Parameter(torch_t.Tensor(n_head, d_model, d_k))
            self.w_vs = nn.Parameter(torch_t.Tensor(n_head, d_model, d_v))

            init.xavier_normal_(self.w_qs)
            init.xavier_normal_(self.w_ks)
            init.xavier_normal_(self.w_vs)

        self.attention = ScaledAttention(hparams, attention_dropout=attention_dropout)
        self.layer_norm = LayerNormalization(d_model)

        if not self.partitioned:
            self.proj = nn.Linear(n_head*d_v, d_model, bias=False)
        else:
            self.proj1 = nn.Linear(n_head*(d_v//2), self.d_content, bias=False)
            self.proj2 = nn.Linear(n_head*(d_v//2), self.d_positional, bias=False)

        self.residual_dropout = FeatureDropout(residual_dropout)

    def split_qkv_packed(self, inp, qk_inp=None):
        v_inp_repeated = inp.repeat(self.n_head, 1).view(self.n_head, -1, inp.size(-1)) # n_head x len_inp x d_model
        if qk_inp is None:
            qk_inp_repeated = v_inp_repeated
        else:
            qk_inp_repeated = qk_inp.repeat(self.n_head, 1).view(self.n_head, -1, qk_inp.size(-1))

        if not self.partitioned:
            q_s = torch.bmm(qk_inp_repeated, self.w_qs) # n_head x len_inp x d_k
            k_s = torch.bmm(qk_inp_repeated, self.w_ks) # n_head x len_inp x d_k
            v_s = torch.bmm(v_inp_repeated, self.w_vs) # n_head x len_inp x d_v
        else:
            q_s = torch.cat([
                torch.bmm(qk_inp_repeated[:,:,:self.d_content], self.w_qs1),
                torch.bmm(qk_inp_repeated[:,:,self.d_content:], self.w_qs2),
                ], -1)
            k_s = torch.cat([
                torch.bmm(qk_inp_repeated[:,:,:self.d_content], self.w_ks1),
                torch.bmm(qk_inp_repeated[:,:,self.d_content:], self.w_ks2),
                ], -1)
            v_s = torch.cat([
                torch.bmm(v_inp_repeated[:,:,:self.d_content], self.w_vs1),
                torch.bmm(v_inp_repeated[:,:,self.d_content:], self.w_vs2),
                ], -1)
        return q_s, k_s, v_s

    def pad_and_rearrange(self, q_s, k_s, v_s, batch_idxs):
        n_head = self.n_head
        d_k, d_v = self.d_k, self.d_v

        len_padded = batch_idxs.max_len
        mb_size = batch_idxs.batch_size
        q_padded = q_s.new_zeros(n_head, mb_size, len_padded, d_k)
        k_padded = k_s.new_zeros(n_head, mb_size, len_padded, d_k)
        v_padded = v_s.new_zeros(n_head, mb_size, len_padded, d_v)

        invalid_mask = q_s.new_ones(mb_size, len_padded).byte().fill_(True)

        for i, (start, end) in enumerate(zip(batch_idxs.boundaries_np[:-1], batch_idxs.boundaries_np[1:])):
            q_padded[:,i,:end-start,:] = q_s[:,start:end,:]
            k_padded[:,i,:end-start,:] = k_s[:,start:end,:]
            v_padded[:,i,:end-start,:] = v_s[:,start:end,:]
            invalid_mask[i, :end-start].fill_(False)

        return(
            q_padded.view(-1, len_padded, d_k),
            k_padded.view(-1, len_padded, d_k),
            v_padded.view(-1, len_padded, d_v),
            invalid_mask.unsqueeze(1).expand(mb_size, len_padded, len_padded).repeat(n_head, 1, 1),
            (~invalid_mask).repeat(n_head, 1),
            )

    def combine_v(self, outputs):
        n_head = self.n_head
        outputs = outputs.view(n_head, -1, self.d_v)

        if not self.partitioned:
            outputs = torch.transpose(outputs, 0, 1).contiguous().view(-1, n_head * self.d_v)

            outputs = self.proj(outputs)
        else:
            d_v1 = self.d_v // 2
            outputs1 = outputs[:,:,:d_v1]
            outputs2 = outputs[:,:,d_v1:]
            outputs1 = torch.transpose(outputs1, 0, 1).contiguous().view(-1, n_head * d_v1)
            outputs2 = torch.transpose(outputs2, 0, 1).contiguous().view(-1, n_head * d_v1)
            outputs = torch.cat([
                self.proj1(outputs1),
                self.proj2(outputs2),
                ], -1)

        return outputs

    def forward(self, inp, batch_idxs, qk_inp=None):
        residual = inp

        q_s, k_s, v_s = self.split_qkv_packed(inp, qk_inp=qk_inp)

        q_padded, k_padded, v_padded, attn_mask, output_mask = self.pad_and_rearrange(q_s, k_s, v_s, batch_idxs)

        outputs_padded, attns_padded = self.attention(
            q_padded, k_padded, v_padded,
            attn_mask=attn_mask
            )
        outputs = outputs_padded[output_mask]
        outputs = self.combine_v(outputs)

        outputs = self.residual_dropout(outputs, batch_idxs)

        return self.layer_norm(outputs + residual), attns_padded

#
class PositionwiseFeedForward(nn.Module):
    """
    A position-wise feed forward module.

    Projects to a higher-dimensional space before applying ReLU, then projects
    back.
    """

    def __init__(self, d_hid, d_ff, relu_dropout=0.1, residual_dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_hid, d_ff)
        self.w_2 = nn.Linear(d_ff, d_hid)

        self.layer_norm = LayerNormalization(d_hid)
        self.relu_dropout = FeatureDropout(relu_dropout)
        self.residual_dropout = FeatureDropout(residual_dropout)
        self.relu = nn.ReLU()


    def forward(self, x, batch_idxs):
        residual = x

        output = self.w_1(x)
        output = self.relu_dropout(self.relu(output), batch_idxs)
        output = self.w_2(output)

        output = self.residual_dropout(output, batch_idxs)
        return self.layer_norm(output + residual)

#
class PartitionedPositionwiseFeedForward(nn.Module):
    def __init__(self, d_hid, d_ff, d_positional, relu_dropout=0.1, residual_dropout=0.1):
        super().__init__()
        self.d_content = d_hid - d_positional
        self.w_1c = nn.Linear(self.d_content, d_ff//2)
        self.w_1p = nn.Linear(d_positional, d_ff//2)
        self.w_2c = nn.Linear(d_ff//2, self.d_content)
        self.w_2p = nn.Linear(d_ff//2, d_positional)
        self.layer_norm = LayerNormalization(d_hid)
        self.relu_dropout = FeatureDropout(relu_dropout)
        self.residual_dropout = FeatureDropout(residual_dropout)
        self.relu = nn.ReLU()

    def forward(self, x, batch_idxs):
        residual = x
        xc = x[:, :self.d_content]
        xp = x[:, self.d_content:]

        outputc = self.w_1c(xc)
        outputc = self.relu_dropout(self.relu(outputc), batch_idxs)
        outputc = self.w_2c(outputc)

        outputp = self.w_1p(xp)
        outputp = self.relu_dropout(self.relu(outputp), batch_idxs)
        outputp = self.w_2p(outputp)

        output = torch.cat([outputc, outputp], -1)

        output = self.residual_dropout(output, batch_idxs)
        return self.layer_norm(output + residual)

#
class MultiLevelEmbedding(nn.Module):
    def __init__(self,
            hparams,
            max_len,
            normalize=True,
            dropout=0.1,
            timing_dropout=0.0,
            extra_content_dropout = 0.2):
        super(MultiLevelEmbedding, self).__init__()
        self.hparams = hparams
        self.partitioned = self.hparams.partitioned

        self.d_embedding = self.hparams.d_model

        if self.partitioned:
            self.d_positional = self.hparams.d_model // 2
            self.d_content = self.d_embedding - self.d_positional
        else:
            self.d_positional = self.d_embedding
            self.d_content = self.d_embedding

        self.extra_content_dropout = FeatureDropout(extra_content_dropout)

        if normalize:
            self.layer_norm = LayerNormalization(self.d_embedding)
        else:
            self.layer_norm = lambda x: x

        self.dropout = FeatureDropout(dropout)
        self.timing_dropout = FeatureDropout(timing_dropout)
        # Learned embeddings
        self.position_table = nn.Parameter(torch_t.FloatTensor(max_len, self.d_positional))
        init.normal_(self.position_table)

    def forward(self, batch_idxs, extra_content_annotations=None):

        content_annotations = self.extra_content_dropout(extra_content_annotations, batch_idxs)
        timing_signal = torch.cat([self.position_table[:seq_len, :] for seq_len in batch_idxs.seq_lens_np], dim=0)
        timing_signal = self.timing_dropout(timing_signal, batch_idxs)
        if self.partitioned:
            annotations = torch.cat([content_annotations, timing_signal], 1)
        else:
            annotations = content_annotations + timing_signal

        annotations = self.layer_norm(self.dropout(annotations, batch_idxs))

        return annotations, content_annotations, batch_idxs

#

def get_xlnet(xlnet_model):
    # Avoid a hard dependency on BERT by only importing it if it's being used
    from pytorch_transformers import (WEIGHTS_NAME, XLNetModel,
                                      XLMConfig, XLMForSequenceClassification,
                                      XLMTokenizer, XLNetConfig, XLNetLMHeadModel,
                                      XLNetForSequenceClassification,
                                      XLNetTokenizer)
    print(xlnet_model)
    tokenizer = XLNetTokenizer.from_pretrained(xlnet_model)
    xlnet = XLNetLMHeadModel.from_pretrained(xlnet_model)

    # if bert_model.endswith('.tar.gz'):
    #     tokenizer = BertTokenizer.from_pretrained(bert_model.replace('.tar.gz', '-vocab.txt'), do_lower_case=bert_do_lower_case)
    # else:
    #     tokenizer = BertTokenizer.from_pretrained(bert_model, do_lower_case=bert_do_lower_case)
    # bert = BertModel.from_pretrained(bert_model)
    return tokenizer, xlnet

# def get_bert(bert_model, bert_do_lower_case):
#     # Avoid a hard dependency on BERT by only importing it if it's being used
#     from pytorch_transformers import BertTokenizer
#     from pytorch_transformers import BertForPreTraining
#     if bert_model.endswith('.tar.gz'):
#
#         tokenizer = BertTokenizer.from_pretrained(bert_model.replace('.tar.gz', '-vocab.txt'), do_lower_case=bert_do_lower_case)
#     else:
#         tokenizer = BertTokenizer.from_pretrained(bert_model, do_lower_case=bert_do_lower_case)
#     bert = BertForPreTraining.from_pretrained(bert_model)
#     return tokenizer, bert

def get_bert(bert_model, bert_do_lower_case, use_albert = False, use_sparse = False, use_electra = False):
    # Avoid a hard dependency on BERT by only importing it if it's being used
    from pytorch_transformers import BertTokenizer, BertModel, BertForPreTraining
    tokenizer = BertTokenizer.from_pretrained(bert_model, do_lower_case=bert_do_lower_case)
    bert = BertForPreTraining.from_pretrained(bert_model, use_albert = use_albert, use_sparse = use_sparse, use_electra = use_electra)
    return tokenizer, bert


class Encoder(nn.Module):
    def __init__(self, hparams, embedding,
                    num_heads=2, d_kv = 32, d_ff=1024,
                    d_positional=None,
                    relu_dropout=0.1, residual_dropout=0.1, attention_dropout=0.1):
        super(Encoder, self).__init__()
        self.emb = embedding
        d_model = embedding.d_embedding
        self.d_model = d_model
        self.hparams = hparams

        d_k = d_v = d_kv

        self.stacks = []

        for i in range(hparams.num_layers):
            attn = MultiHeadAttention(hparams, num_heads, d_model, d_k, d_v, residual_dropout=residual_dropout,
                                      attention_dropout=attention_dropout, d_positional=d_positional)
            if d_positional is None:
                ff = PositionwiseFeedForward(d_model, d_ff, relu_dropout=relu_dropout,
                                             residual_dropout=residual_dropout)
            else:
                ff = PartitionedPositionwiseFeedForward(d_model, d_ff, d_positional, relu_dropout=relu_dropout,
                                                        residual_dropout=residual_dropout)

            self.add_module(f"attn_{i}", attn)
            self.add_module(f"ff_{i}", ff)

            self.stacks.append((attn, ff))

    def forward(self, batch_idxs, extra_content_annotations=None):
        res, res_c, batch_idxs = self.emb(batch_idxs, extra_content_annotations=extra_content_annotations)

        for i, (attn, ff) in enumerate(self.stacks):
            res, current_attns = attn(res, batch_idxs)
            res = ff(res, batch_idxs)

        return res, batch_idxs

class Jointmodel(nn.Module):
    def __init__(
            self,
            tag_vocab,
            word_vocab,
            label_vocab,
            char_vocab,
            type_vocab,
            srl_vocab,
            hparams,
    ):
        super(Jointmodel,self).__init__()
        self.spec = locals()
        self.spec.pop("self")
        self.spec.pop("__class__")
        self.spec['hparams'] = hparams.to_dict()

        self.tag_vocab = tag_vocab
        self.word_vocab = word_vocab
        self.label_vocab = label_vocab
        self.char_vocab = char_vocab
        self.type_vocab = type_vocab
        self.srl_vocab = srl_vocab

        self.hparams = hparams
        self.d_model = hparams.d_model

        self.partitioned = hparams.partitioned
        # self.d_content = (self.d_model // 2) if self.partitioned else self.d_model
        # self.d_positional = (hparams.d_model // 2) if self.partitioned else None

        self.morpho_emb_dropout = hparams.morpho_emb_dropout
        self.d_content = self.d_model

        if self.hparams.model =="xlnet":

            self.tokenizer, self.xlnet = get_xlnet(hparams.bert_model)

            d_xlnet_annotations = self.xlnet.transformer.d_model
            self.xlnet_max_len = 512

            self.project_xlnet = nn.Linear(d_xlnet_annotations, self.d_content, bias=False)
        elif self.hparams.model =="bert":

            self.tokenizer, self.bert = get_bert(hparams.bert_model, hparams.bert_do_lower_case, use_sparse = self.hparams.use_sparse, use_electra = self.hparams.use_electra)

            d_bert_annotations = self.bert.config.hidden_size
            self.project_bert = nn.Linear(d_bert_annotations, self.d_content, bias=False)

        elif self.hparams.model =="albert":

            self.tokenizer, self.bert = get_bert(hparams.bert_model, hparams.bert_do_lower_case, use_albert=True, use_sparse = self.hparams.use_sparse, use_electra = self.hparams.use_electra)

            d_bert_annotations = self.bert.config.hidden_size
            self.project_bert = nn.Linear(d_bert_annotations, self.d_content, bias=False)

        # if hparams.use_only_bert:
        #
        #     self.d_content = self.d_model
        # else:
        #
        #     self.embedding = MultiLevelEmbedding(
        #         hparams=hparams,
        #         max_len = self.bert_max_len,
        #         dropout=hparams.embedding_dropout,
        #         timing_dropout=hparams.timing_dropout,
        #         extra_content_dropout=self.morpho_emb_dropout,
        #
        #     )
        #
        #     self.encoder = Encoder(
        #         hparams,
        #         self.embedding,
        #         num_heads=hparams.num_heads,
        #         d_kv=hparams.d_kv,
        #         d_ff=hparams.d_ff,
        #         d_positional=self.d_positional,
        #         relu_dropout=hparams.relu_dropout,
        #         residual_dropout=hparams.residual_dropout,
        #         attention_dropout=hparams.attention_dropout,
        #     )



        self.decoder = Uniform_Decoder(
            tag_vocab,
            word_vocab,
            label_vocab,
            char_vocab,
            type_vocab,
            srl_vocab,
            hparams
        )

        # if use_cuda:
        #     self.cuda()

    @property
    def model(self):
        return self.state_dict()

    @classmethod
    def from_spec(cls, spec, model):
        spec = spec.copy()
        hparams = spec['hparams']
        if 'joint_pos' not in hparams:
            hparams['joint_pos'] = False
        if 'use_xlnet' not in hparams:
            hparams['use_xlnet'] = False
        if 'model' not in hparams:
            hparams['model'] = "bert"
        hparams['bert_model'] = "bert-large-uncased-whole-word-masking"

        spec['hparams'] = makehp.HParams(**hparams)
        res = cls(**spec)
        # if use_cuda:
        #     res.cpu()
        res.load_state_dict(model)
        # if use_cuda:
        #     res.cuda()
        return res

    def split_batch(self, sentences, golds, srlspans, srldeps, subbatch_max_tokens=3000):
        lens = [len(sentence) + 2 for sentence in sentences]

        lens = np.asarray(lens, dtype=int)
        lens_argsort = np.argsort(lens).tolist()

        num_subbatches = 0
        subbatch_size = 1
        while lens_argsort:
            if (subbatch_size == len(lens_argsort)) or (subbatch_size * lens[lens_argsort[subbatch_size]] > subbatch_max_tokens):
                yield [sentences[i] for i in lens_argsort[:subbatch_size]], [golds[i] for i in lens_argsort[:subbatch_size]], \
                      [srlspans[i] for i in lens_argsort[:subbatch_size]], [srldeps[i] for i in lens_argsort[:subbatch_size]]
                lens_argsort = lens_argsort[subbatch_size:]
                num_subbatches += 1
                subbatch_size = 1
            else:
                subbatch_size += 1

    def forward(self, sentences, gold_trees=None, gold_srlspans=None, gold_srldeps=None, gold_verbs=None, bert_data = None):
        is_train = gold_trees is not None
        self.train(is_train)
        torch.set_grad_enabled(is_train)
        if is_train:
            dis_idx, input_ids, origin_ids, input_mask, word_start_mask, word_end_mask, segment_ids, perm_mask, target_mapping, lm_label_ids, lm_label_mask, is_next = bert_data
        else:
            dis_idx, input_ids, input_mask, word_start_mask, word_end_mask, segment_ids, lm_label_ids, is_next = bert_data
            perm_mask = None
            target_mapping = None


        sentences = [sentences[idx] for idx in dis_idx]

        if gold_trees is None:
            gold_trees = [None] * len(sentences)
            gold_srlspans = [None] * len(sentences)
            gold_srldeps = [None] * len(sentences)
        else:
            gold_trees = [gold_trees[idx] for idx in dis_idx]
            gold_srlspans = [gold_srlspans[idx] for idx in dis_idx]
            gold_srldeps = [gold_srldeps[idx] for idx in dis_idx]

        # word_start_mask = torch.tensor(word_start_mask)
        #  For now, just project the features from the last word piece in each word
        if self.hparams.model =="xlnet":
            lm_loss, features = self.xlnet(input_ids, token_type_ids = segment_ids, attention_mask = input_mask, perm_mask=perm_mask, target_mapping=target_mapping, labels = lm_label_ids)
            features_packed = features.masked_select(word_start_mask.to(torch.uint8).unsqueeze(-1)).reshape(-1,features.shape[-1])
            extra_content_annotations = self.project_xlnet(features_packed)
        else:
            lm_loss, features, prediction_ids, all_syntax_att = self.bert(input_ids, token_type_ids = segment_ids, attention_mask = input_mask, masked_lm_labels = lm_label_ids, next_sentence_label = is_next)

            if self.hparams.use_electra and is_train:
                if self.hparams.use_alltoken:
                    prediction_ids.masked_fill_((1 - input_mask).byte(), 0)
                    discrinput_ids = prediction_ids
                else:
                    lm_label_mask = lm_label_mask.byte()
                    prediction_ids_tmp = prediction_ids.clone().detach().requires_grad_(False)
                    prediction_ids_tmp = torch.masked_select(prediction_ids_tmp, lm_label_mask)
                    input_ids_tmp = input_ids.clone().detach().requires_grad_(False)
                    input_ids_tmp.masked_scatter_(lm_label_mask, prediction_ids_tmp)
                    discrinput_ids = input_ids_tmp

                discr_labels = torch.eq(origin_ids, discrinput_ids).long()
                discr_labels.masked_fill_((1 - input_mask).byte(), -1)
                # print("origin_ids", origin_ids[0])
                # print("discrinput_ids:",discrinput_ids[0])
                # print("discr_labels",discr_labels[0])
                # print("lm_label_ids", lm_label_ids[0])
                discr_loss, features, prediction_ids, all_syntax_att = self.bert(discrinput_ids, token_type_ids=segment_ids,
                                                                              attention_mask=input_mask, is_discr = True,
                                                                              original_labels=discr_labels)
                lm_loss = lm_loss + 50.0 * discr_loss

            features_packed = features.masked_select(word_start_mask.to(torch.uint8).unsqueeze(-1)).reshape(-1,features.shape[-1])
            #print(features_packed.size())
            extra_content_annotations = self.project_bert(features_packed)

        packed_len = sum([(len(sentence) + 2) for sentence in sentences])

        i = 0
        batch_idxs = np.zeros(packed_len, dtype=int)
        for snum, sentence in enumerate(sentences):
            # print(sentence)
            for (tag, word) in [(START, START)] + sentence + [(STOP, STOP)]:
                batch_idxs[i] = snum
                i += 1
        assert i == packed_len

        batch_idxs = BatchIndices(batch_idxs)

        assert extra_content_annotations.size(0) == packed_len

        annotations = extra_content_annotations

        # if self.hparams.use_only_bert:
        #     annotations = extra_content_annotations
        # else:
        #     annotations, _ = self.encoder(batch_idxs, extra_content_annotations=extra_content_annotations)

            # if self.partitioned:
            #     annotations = torch.cat([
            #         annotations[:, 0::2],
            #         annotations[:, 1::2],
            #     ], 1)

        fencepost_annotations = torch.cat([
            annotations[:-1, :self.d_model // 2],
            annotations[1:, self.d_model // 2:],
        ], 1)

        fencepost_annotations_start = fencepost_annotations
        fencepost_annotations_end = fencepost_annotations


        if not is_train:

            decoder_args = dict(
                fencepost_annotations_start=fencepost_annotations_start,
                fencepost_annotations_end=fencepost_annotations_end,
                batch_idxs=batch_idxs,
                sentences=sentences,
                gold_verbs = gold_verbs
            )

            syntree_pred, score_list, srlspan_pred, srldep_pred = self.decoder.decode(annotations, **decoder_args)

            return syntree_pred, srlspan_pred, srldep_pred

            # pos_pred = [json.dumps([leaf.goldtag for leaf in tree.leaves()]) for tree in syntree_pred]
            # syntree_linearized = [tree.linearize() for tree in syntree_pred]
            # dev_pred_head = [json.dumps([str(leaf.father) for leaf in tree.leaves()]) for tree in syntree_pred]
            # dev_pred_type = [json.dumps([leaf.type for leaf in tree.leaves()]) for tree in syntree_pred]
            #
            # srlspan_strlist = []
            # for srlspan in srlspan_pred:
            #     srlspan_str = {}
            #     for pred_id, args in srlspan.items():
            #         srlspan_str[str(pred_id)] = [(str(a[0]), str(a[1]), a[2]) for a in args]
            #     srlspan_strlist.append(json.dumps(srlspan_str))
            # srldep_strlist = []
            # for srldep in srldep_pred:
            #     srldep_str = {}
            #     for pred_id, args in srldep.items():
            #         srldep_str[str(pred_id)] = [(str(a[0]), a[1]) for a in args]
            #     srldep_strlist.append(json.dumps(srldep_str))


            # return syntree_linearized, dev_pred_head, dev_pred_type, srlspan_strlist, srldep_strlist, pos_pred
        else:

            loss_args = dict(
                fencepost_annotations_start = fencepost_annotations_start,
                fencepost_annotations_end =fencepost_annotations_end,
                batch_idxs =batch_idxs,
                sentences =sentences,
                gold_trees = gold_trees,
                gold_srlspans = gold_srlspans,
                gold_srldeps = gold_srldeps)

            task_loss, srl_loss, synconst_loss, syndep_loss = self.decoder.cal_loss(annotations, **loss_args)

            #loss = bert_loss + task_loss

            return lm_loss, task_loss