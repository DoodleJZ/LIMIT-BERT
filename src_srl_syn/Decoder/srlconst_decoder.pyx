import numpy as np
cimport numpy as np
from numpy cimport ndarray
cimport cython

ctypedef np.float32_t DTYPE_t

ORACLE_PRECOMPUTED_TABLE = {}

@cython.boundscheck(False)
def decode(int force_gold, int sentence_len, np.ndarray[DTYPE_t, ndim=3] label_scores_chart, np.ndarray[DTYPE_t, ndim=3] srl_label_chart,
           np.ndarray[int, ndim=2] srlspan_mat, int is_train, gold, label_vocab, srlspan_vocab, np.ndarray[int, ndim=1] prd_verb_id, np.ndarray[int, ndim=1] verb_mask):

    cdef DTYPE_t NEG_INF = -np.inf

    # Label scores chart is copied so we can modify it in-place for augmentated decode
    cdef np.ndarray[DTYPE_t, ndim=3] label_scores_chart_copy = label_scores_chart.copy()
    cdef np.ndarray[DTYPE_t, ndim=3] srl_label_chart_copy = srl_label_chart.copy()
    cdef np.ndarray[DTYPE_t, ndim=2] value_chart = np.zeros((sentence_len+1, sentence_len+1), dtype=np.float32)
    cdef np.ndarray[int, ndim=2] split_idx_chart = np.zeros((sentence_len+1, sentence_len+1), dtype=np.int32)
    cdef np.ndarray[int, ndim=2] best_label_chart = np.zeros((sentence_len+1, sentence_len+1), dtype=np.int32)
    cdef np.ndarray[int, ndim=2] best_srlspan_chart = np.zeros((srl_label_chart.shape[0], srl_label_chart.shape[1]), dtype=np.int32)

    cdef int length
    cdef int left
    cdef int right

    cdef np.ndarray[DTYPE_t, ndim=1] label_scores_for_span

    cdef int oracle_label_index
    cdef DTYPE_t label_score, empty_score, span_score
    cdef DTYPE_t srlspan_score_verb
    cdef int argmax_label_index
    cdef DTYPE_t left_score
    cdef DTYPE_t right_score

    cdef int best_split
    cdef int split_idx # Loop variable for splitting
    cdef DTYPE_t split_val # best so far
    cdef DTYPE_t max_split_val

    cdef int label_index_iter, srlspan_index_iter, gold_verb_idx, pred_verb_idx
    cdef int srlspan_id, srlspan_verb_index, gold_verb_len
    cdef DTYPE_t srlspan_score
    cdef int argmax_srlspan_index

    cdef int cun, g
    cdef np.ndarray[int, ndim=2] oracle_label_chart
    cdef np.ndarray[int, ndim=3] oracle_srlabel_chart
    cdef np.ndarray[int, ndim=2] oracle_split_chart
    cdef np.ndarray[int, ndim=1] gold_verb_id

    cdef int num_verb = srl_label_chart.shape[1]

    if is_train or force_gold:

        if gold not in ORACLE_PRECOMPUTED_TABLE:
            oracle_label_chart = np.zeros((sentence_len+1, sentence_len+1), dtype=np.int32)
            oracle_srlabel_chart = np.zeros((sentence_len+1, sentence_len+1, 50), dtype=np.int32)
            oracle_split_chart = np.zeros((sentence_len+1, sentence_len+1), dtype=np.int32)
            gold_verb_id = -np.ones((sentence_len+1), dtype=np.int32)   #idx of gold verb of each word
            gold_verb_len = 0
            #gold verb: [0,2,3,4] #sorted
            #gold.leaves(): [1,0,1,1,1,0]
            #gold_verb_id: [0,-1,1,2,3,-1] len = 6 = n

            for g, leaf in enumerate(gold.leaves()):    #gold.leaves() len = n, without <start>
                if leaf.verb == 1:
                    gold_verb_id[gold_verb_len] = g           #gold verb idx from 0
                    gold_verb_len = gold_verb_len + 1
                # print(leaf.word, leaf.verb, leaf.srlspan)

            # print(gold_verb_id)
            # print(gold_verb_len)

            for length in range(1, sentence_len + 1):
                for left in range(0, sentence_len + 1 - length):
                    right = left + length
                    oracle_label_chart[left, right] = label_vocab.index(gold.oracle_label(left, right))

                    srlabel_list = gold.oracle_srlspan(left, right)

                    if srlabel_list is not None:   #is None, then oracle_srlspan is '*'
                        # print(gold.chil_enclosing(left,right).words, srlabel_list)
                        for g in range(len(srlabel_list)):
                            oracle_srlabel_chart[left, right, g] = srlspan_vocab.index(srlabel_list[g])
                    if length == 1:
                        continue
                    oracle_splits = gold.oracle_splits(left, right)
                    oracle_split_chart[left, right] = min(oracle_splits)
            if not gold.nocache:
                ORACLE_PRECOMPUTED_TABLE[gold] = oracle_label_chart, oracle_srlabel_chart, gold_verb_id, gold_verb_len, oracle_split_chart
        else:
            oracle_label_chart, oracle_srlabel_chart, gold_verb_id, gold_verb_len, oracle_split_chart = ORACLE_PRECOMPUTED_TABLE[gold]


    for length in range(1, sentence_len + 1):
        for left in range(0, sentence_len + 1 - length):
            right = left + length
            srlspan_id = srlspan_mat[left, right]

            if is_train or force_gold:
                oracle_label_index = oracle_label_chart[left, right]

            if force_gold:
                label_score = label_scores_chart_copy[left, right, oracle_label_index]
                best_label_chart[left, right] = oracle_label_index

                #prd_verb_id start from 0, contain <start>
                #prd_verb_id : [5,3,0,1,4]
                #gold verb: [0,2,3,4] #sorted
                #gold.leaves(): [1,0,1,1,1,0]
                #gold_verb_id: [0,-1,1,2,3,-1] len = 6 = n

                srlspan_score = 0

                if srlspan_id != -1:
                    for srlspan_verb_index in range(gold_verb_len):
                        pred_verb_idx = verb_mask[gold_verb_id[srlspan_verb_index]]
                        if pred_verb_idx != -1:
                            best_srlspan_chart[srlspan_id, pred_verb_idx] = oracle_srlabel_chart[left, right, srlspan_verb_index]
                            srlspan_score += srl_label_chart_copy[srlspan_id, pred_verb_idx, best_srlspan_chart[srlspan_id, pred_verb_idx]]
                            # if best_srlspan_chart[srlspan_id, pred_verb_idx] > 0:
                            #     print(left, right, srlspan_vocab.value(best_srlspan_chart[srlspan_id, pred_verb_idx]), gold.chil_enclosing(left,right).words)

                span_score = srlspan_score + label_score
            else:

                if is_train:
                    # augment: here we subtract 1 from the oracle label
                    label_scores_chart_copy[left, right, oracle_label_index] -= 1

                    # srl merge
                    # if srlspan_id != -1:
                    #     for srlspan_verb_index in range(gold_verb_len):
                    #         pred_verb_idx = verb_mask[gold_verb_id[srlspan_verb_index]]
                    #         if gold_verb_idx != -1:
                    #             srl_label_chart_copy[srlspan_id, pred_verb_idx, oracle_srlabel_chart[left, right, srlspan_verb_index]] -= 1


                # We do argmax ourselves to make sure it compiles to pure C
                # if length < sentence_len:
                #     argmax_label_index = 0
                # else:
                #     # Not-a-span label is not allowed at the root of the tree
                #     argmax_label_index = 1

                argmax_label_index = 1  #not empty

                label_score = label_scores_chart_copy[left, right, argmax_label_index]
                for label_index_iter in range(1, label_scores_chart_copy.shape[2]):
                    if label_scores_chart_copy[left, right, label_index_iter] > label_score:
                        argmax_label_index = label_index_iter
                        label_score = label_scores_chart_copy[left, right, label_index_iter]
                best_label_chart[left, right] = argmax_label_index

                srlspan_score = 0
                srlspan_empty = 0

                if srlspan_id != -1 :

                    for srlspan_verb_index in range(srl_label_chart_copy.shape[1]):

                        srlspan_score_verb = srl_label_chart_copy[srlspan_id, srlspan_verb_index, 0]
                        argmax_srlspan_index = 0
                        srlspan_score_verb_empty = srl_label_chart_copy[srlspan_id, srlspan_verb_index, 0]

                        for srlspan_index_iter in range(srl_label_chart_copy.shape[2]):
                            if srl_label_chart_copy[srlspan_id, srlspan_verb_index, srlspan_index_iter] > srlspan_score_verb:
                                argmax_srlspan_index = srlspan_index_iter
                                srlspan_score_verb = srl_label_chart_copy[srlspan_id, srlspan_verb_index, srlspan_index_iter]

                        # if is_train:
                        #     srlspan_score_verb += 1     # augment: here we add 1 to all srlspan scores in the mat
                        #
                        #     srlspan_score_verb_empty += 1

                        srlspan_score += srlspan_score_verb
                        srlspan_empty += srlspan_score_verb_empty
                        best_srlspan_chart[srlspan_id, srlspan_verb_index] = argmax_srlspan_index

                if is_train:
                    # augment: here we add 1 to all label scores
                    label_score += 1

                span_score = label_score + srlspan_score        #not empty
                empty_score = label_scores_chart_copy[left, right, 0] + 1

                if length == 1:
                    empty_score = empty_score + srlspan_score # leaf empty label can have srl label
                else:
                    empty_score = empty_score + srlspan_empty

                if span_score < empty_score:    #empty span, srl all is '*'
                    if length < sentence_len:
                        #empty label
                        best_label_chart[left, right] = 0
                        span_score = empty_score
                        if srlspan_id != -1 :
                            for srlspan_verb_index in range(srl_label_chart_copy.shape[1]):
                                best_srlspan_chart[srlspan_id, srlspan_verb_index] = 0

            if length == 1:

                value_chart[left, right] = span_score

                continue

            if force_gold:
                best_split = oracle_split_chart[left, right]
            else:
                best_split = left + 1
                split_val = NEG_INF
                for split_idx in range(left + 1, right):
                    max_split_val = value_chart[left, split_idx] + value_chart[split_idx, right]
                    if max_split_val > split_val:
                        split_val = max_split_val
                        best_split = split_idx

            value_chart[left, right] = span_score + value_chart[left, best_split] + value_chart[best_split, right]
            split_idx_chart[left, right] = best_split

    # Now we need to recover the tree by traversing the chart starting at the
    # root. This iterative implementation is faster than any of my attempts to
    # use helper functions and recursion

    # All fully binarized trees have the same number of nodes
    cdef int num_tree_nodes = 2 * sentence_len - 1
    cdef np.ndarray[int, ndim=1] included_i = np.empty(num_tree_nodes, dtype=np.int32)
    cdef np.ndarray[int, ndim=1] included_j = np.empty(num_tree_nodes, dtype=np.int32)
    cdef np.ndarray[int, ndim=1] included_label = np.empty(num_tree_nodes, dtype=np.int32)
    cdef np.ndarray[int, ndim=2] included_srlspan = np.zeros((num_tree_nodes, 50), dtype=np.int32)
    cdef np.ndarray[int, ndim=3] included_srlspan_mask = np.zeros((srl_label_chart.shape[0], srl_label_chart.shape[1], srl_label_chart.shape[2]), dtype=np.int32)

    cdef int idx = 0
    cdef int stack_idx = 1
    # technically, the maximum stack depth is smaller than this
    cdef np.ndarray[int, ndim=1] stack_i = np.empty(num_tree_nodes + 5, dtype=np.int32)
    cdef np.ndarray[int, ndim=1] stack_j = np.empty(num_tree_nodes + 5, dtype=np.int32)
    stack_i[1] = 0
    stack_j[1] = sentence_len

    cdef int i, j, k

    while stack_idx > 0:
        i = stack_i[stack_idx]
        j = stack_j[stack_idx]
        stack_idx -= 1
        included_i[idx] = i
        included_j[idx] = j
        included_label[idx] = best_label_chart[i, j]

        srlspan_id = srlspan_mat[i, j]
        if srlspan_id != -1:
            for srlspan_verb_index in range(srl_label_chart.shape[1]):
                included_srlspan_mask[srlspan_id, srlspan_verb_index, best_srlspan_chart[srlspan_id, srlspan_verb_index]] = 1
                included_srlspan[idx, srlspan_verb_index] = best_srlspan_chart[srlspan_id, srlspan_verb_index]

        idx += 1
        if i + 1 < j:
            k = split_idx_chart[i, j]
            stack_idx += 1
            stack_i[stack_idx] = k
            stack_j[stack_idx] = j
            stack_idx += 1
            stack_i[stack_idx] = i
            stack_j[stack_idx] = k

    cdef DTYPE_t running_total = 0.0
    for idx in range(num_tree_nodes):
        running_total += label_scores_chart[included_i[idx], included_j[idx], included_label[idx]]
        srlspan_id = srlspan_mat[included_i[idx], included_j[idx]]
        if srlspan_id != -1:
            for srlspan_verb_index in range(srl_label_chart.shape[1]):
                running_total += srl_label_chart[srlspan_id, srlspan_verb_index, included_srlspan[idx, srlspan_verb_index]]

    cdef DTYPE_t score = value_chart[0, sentence_len]
    cdef DTYPE_t augment_amount = round(score - running_total)

    return score, included_i.astype(int), included_j.astype(int), included_label.astype(int), included_srlspan_mask.astype(int), augment_amount
