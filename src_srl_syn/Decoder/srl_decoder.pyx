import numpy as np
cimport numpy as np
from numpy cimport ndarray
cimport cython

ctypedef np.float32_t DTYPE_t

_CORE_ARGS = {"A0": 1, "A1": 2, "A2": 4, "A3": 8, "A4": 16, "A5": 32, "AA": 64}
_CORE_ARGS_INDEX = {"A0": 0, "A1": 1, "A2": 2, "A3": 3, "A4": 4, "A5": 5, "AA": 6}
_CORE_ARGS_LIST = ["A0", "A1", "A2", "A3", "A4", "A5", "AA"]

@cython.boundscheck(False)
def decode(int start, int end, int leng, int verb_idx, np.ndarray[int, ndim=2] span_mat, np.ndarray[DTYPE_t, ndim=2] srl_score_np, srlspan_dict, srldep_dict, srl_vocab):

    cdef DTYPE_t NEG_INF = -np.inf, span_score, max_score, max_dep_score
    cdef int cun, left, right, state, span_idx, cur, cur_state, dep_label_idx, span_label_idx, dep_span_idx, dep_idx, i
    cdef np.ndarray[DTYPE_t, ndim=1] max_srlscore = np.zeros(leng, dtype= np.float32)
    cdef np.ndarray[int, ndim=1] srl_path = np.zeros(leng, dtype= np.int32)
    cdef np.ndarray[DTYPE_t, ndim=2] srldep_label_score = np.zeros((leng, srl_score_np.shape[1]), dtype= np.float32)

    for right in range(start + 1, end):
        max_score = -np.inf
        srldep_label_score[right,:] = srldep_label_score[right - 1,:]
        dep_span_idx = span_mat[right][right]
        if dep_span_idx != -1:
            for i in range(1, srl_score_np.shape[1]):  #without empty label
                srl_label = srl_vocab.value(i)      #span srl label
                if srl_label[-1]=="*":
                    srl_label = srl_label[:-1]
                    srl_label_al = srl_label
                else:
                    if srl_label[0] == "C":
                        srl_label_al = srl_label[2:]    #without C
                    else:
                        srl_label_al = "C-" + srl_label   #with C

                max_dep_score = srl_score_np[dep_span_idx, srl_vocab.index(srl_label)]
                if srl_label_al in srl_vocab.values and max_dep_score < srl_score_np[dep_span_idx, srl_vocab.index(srl_label_al)]:
                    max_dep_score = srl_score_np[dep_span_idx, srl_vocab.index(srl_label_al)]
                if max_dep_score > srl_score_np[dep_span_idx, 0]:
                    srldep_label_score[right, i] += max_dep_score

        for left in range(start, right):
            span_idx = span_mat[left][right]
            if span_idx == -1:
                span_score = np.max(srldep_label_score[right] - srldep_label_score[left])
            else:
                span_score = np.max(srl_score_np[span_idx] + srldep_label_score[right] - srldep_label_score[left])

            if span_score + max_srlscore[left] > max_score:
                max_score = span_score + max_srlscore[left]
                srl_path[right] = left

        max_srlscore[right] = max_score

    cur = end - 1
    while cur > start:
        left = srl_path[cur]
        right = cur
        span_idx = span_mat[left][right]
        if span_idx != -1:
            span_label_idx = np.argmax(srl_score_np[span_idx] + srldep_label_score[right] - srldep_label_score[left])
        else:
            span_label_idx = np.argmax(srldep_label_score[right] - srldep_label_score[left])
        if span_label_idx > 0:
            if srl_vocab.value(span_label_idx)[-1] != "*":
                if verb_idx not in srlspan_dict:
                    srlspan_dict[verb_idx] = []
                srlspan_dict[verb_idx].append((left, right - 1, srl_vocab.value(span_label_idx)))
            for dep_idx in range(left + 1, right + 1):
                dep_span_idx = span_mat[dep_idx][dep_idx]
                if dep_span_idx != -1:
                    srl_label = srl_vocab.value(span_label_idx)
                    if srl_label[0] == "V":
                        continue
                    if srl_label[-1]=="*":      # just contain dep srl
                        srl_label = srl_label[:-1]
                        srl_label_al = srl_label
                    else:                       # srlspan label, C- srlspan label, srlspan label*
                        if srl_label[0] == "C":
                            srl_label_al = srl_label[2:]
                        else:
                            srl_label_al = "C-" + srl_label

                    max_dep_score = srl_score_np[dep_span_idx, srl_vocab.index(srl_label) ]
                    if srl_label_al in srl_vocab.values and max_dep_score < srl_score_np[dep_span_idx, srl_vocab.index(srl_label_al) ]:
                        max_dep_score = srl_score_np[dep_span_idx, srl_vocab.index(srl_label_al) ]
                        srl_label = srl_label_al
                    if max_dep_score > 0:  #better than empty
                        if verb_idx not in srldep_dict:
                            srldep_dict[verb_idx] = []
                        srldep_dict[verb_idx].append((dep_idx - 1, srl_label))      #start from 0

        cur = left
