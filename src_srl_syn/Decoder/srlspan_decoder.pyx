import numpy as np
cimport numpy as np
from numpy cimport ndarray
cimport cython

ctypedef np.float32_t DTYPE_t

_CORE_ARGS = {"A0": 1, "A1": 2, "A2": 4, "A3": 8, "A4": 16, "A5": 32, "AA": 64}
_CORE_ARGS_INDEX = {"A0": 0, "A1": 1, "A2": 2, "A3": 3, "A4": 4, "A5": 5, "AA": 6}
_CORE_ARGS_LIST = ["A0", "A1", "A2", "A3", "A4", "A5", "AA"]

@cython.boundscheck(False)
def decode(int leng, np.ndarray[int, ndim=1] verb_id_np, np.ndarray[int, ndim=2] span_mat, np.ndarray[DTYPE_t, ndim=3] srl_score_np, srlspan_vocab):

    cdef DTYPE_t NEG_INF = -np.inf, span_score, max_score
    cdef int cun, left, right, state, span_idx, cur, cur_state, max_label_idx, label_idx

    cdef np.ndarray[DTYPE_t, ndim=2] max_srlscore = np.zeros((leng,128), dtype=np.float32)
    for state in range(1,128):
        max_srlscore[0, state] = NEG_INF
    cdef np.ndarray[int, ndim=2] srl_path = np.zeros((leng,128), dtype= np.int32)
    cdef np.ndarray[int, ndim=2] srl_state = np.zeros((leng,128), dtype= np.int32)
    cdef np.ndarray[int, ndim=1] core_mask = np.zeros(10, dtype= np.int32)
    srl_dpdict = {}
    for j, verb_idx in enumerate(verb_id_np):
        for right in range(1, leng):
            for state in range(128):    #total state in binary
                max_score = NEG_INF
                cur_state = state
                core_mask = np.zeros(10, dtype= np.int32)
                cun = 0
                while cur_state > 0:
                    if cur_state % 2 == 1:
                        core_mask[cun] = 1
                    cur_state = cur_state // 2
                    cun += 1

                for left in range(right):

                    span_idx = span_mat[left][right]
                    if span_idx == -1:
                        span_score = 0
                        if span_score + max_srlscore[left][state] > max_score:
                            max_score = span_score + max_srlscore[left][state]
                            srl_path[right][state] = left
                            srl_state[right][state] = 0
                    else:
                        for arg_index, cur_arg in enumerate(srlspan_vocab.values):
                            span_score = srl_score_np[span_idx, j, arg_index]
                            if cur_arg not in _CORE_ARGS_LIST:        #other label
                                if span_score + max_srlscore[left][state] > max_score:
                                    max_score = span_score + max_srlscore[left][state]
                                    srl_path[right][state] = left
                                    srl_state[right][state] = arg_index
                            elif core_mask[_CORE_ARGS_INDEX[cur_arg]] == 1:           #core label
                                if span_score + max_srlscore[left][state - _CORE_ARGS[cur_arg]] > max_score:
                                    max_score = span_score + max_srlscore[left][state - _CORE_ARGS[cur_arg]]
                                    srl_path[right][state] = left
                                    srl_state[right][state] = arg_index

                max_srlscore[right][state] = max_score

        cur_state = np.argmax(max_srlscore[leng-1])
        cur = leng - 1
        while cur > 0:
            #print(cur)
            left = srl_path[cur][cur_state]
            right = cur
            span_idx = span_mat[left][right]
            if span_idx != -1:
                max_label_idx = srl_state[cur][cur_state]
                if max_label_idx > 0:
                    srlspan_label = srlspan_vocab.value(max_label_idx)
                    if verb_idx not in srl_dpdict:
                        srl_dpdict[verb_idx] = []
                    srl_dpdict[verb_idx].append((left, right, srlspan_label))
                    if srlspan_label in _CORE_ARGS_LIST:
                        cur_state = cur_state - _CORE_ARGS[srlspan_label]

            cur = left

    return srl_dpdict
