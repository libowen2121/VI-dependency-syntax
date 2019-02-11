from Tree import tree
import time

SHIFT = 0
REDUCE_L = 1
REDUCE_R = 2

def check_projective(heads):
    """
    brute force check non-projectivity
    """ 
    if len(heads) < 1:
        raise ValueError('length of heads should be larger than 0')
    if not isinstance(heads[0], int):
        heads = [int(head) for head in heads]
    arcs = [ (heads[mod], mod+1) for mod in range(len(heads)) ]
    for p in range(len(arcs)):
        for q in range(len(arcs)):
            if p != q:
                a = min(arcs[p])
                b = max(arcs[p])
                c = min(arcs[q])
                d = max(arcs[q])
                if a in range(c + 1, d) and b > d or b in range(c + 1, d) and a < c:
                    return False
    return True

def get_dep_oracle(heads):
    """
    get shift-reduce actions
    """
    if len(heads) < 1:
        raise ValueError('length of heads should be larger than 0')
    if not isinstance(heads[0], int):
        heads = [int(head) for head in heads]
    arcs = [(heads[mod], mod + 1) for mod in range(len(heads))]
    action = []
    t = tree(range(len(heads) + 1), arcs)

    buffer = range(1, len(arcs) + 1)
    buffer.reverse()
    stack = []

    while not (len(stack) == 1 and len(buffer) == 0):
        if len(stack) < 2:
            # shift
            stack.append(buffer.pop())
            action.append(SHIFT)
        else:
            i = stack[-2]
            j = stack[-1]
            child = None
            if (i, j) in arcs and t.remove_node(j):
                # reduce_r
                action.append(REDUCE_R)
                child = j
                stack.remove(child)
                continue
            if (j, i) in arcs and t.remove_node(i):
                # reduce_l
                action.append(REDUCE_L)
                child = i
                stack.remove(child)
                continue
            if not child:
                # shift
                if len(buffer) == 0:
                    # non projective dependency tree
                    raise ValueError('Encounter a non-projective/non-single-head tree')
                stack.append(buffer.pop())
                action.append(SHIFT)
    assert len(heads) * 2 - 1 == len(action)
    return action

def read_cluster(cluster_fname, w2i_word):
    """
    Arguments:
        cluster_fname(str):
        w2i_word (dictionary):
    """
    wordi2ci = {}   # word 2 cluster idx
    wordi2i = {}    # word 2 intra cluster idx
    cw = set()
    ci2wordi = {}  # cluster idx 2 word idx list
    c_set = set()
    c_list = []
    with open(cluster_fname) as f:
        for line in f:
            binary, word, _ = line.split()
            c = int(binary, 2)
            if c not in c_set:
                c_list.append(len(c_list))
                ci2wordi[c_list[-1]] = []
                c_set.add(c)
            if word in w2i_word:
                wordi2i[w2i_word[word]] = len(ci2wordi[c_list[-1]])
                ci2wordi[c_list[-1]].append(w2i_word[word])
                wordi2ci[w2i_word[word]] = c_list[-1]
            cw.add(word)

    extra = set(w2i_word.keys()) - cw
    if len(extra) > 0:  # add one more cluster for extra words
        c_list.append(len(c_list))
        ci2wordi[c_list[-1]] = []
        for word in extra:
            wordi2i[w2i_word[word]] = len(ci2wordi[c_list[-1]])
            ci2wordi[c_list[-1]].append(w2i_word[word])
            wordi2ci[w2i_word[word]] = c_list[-1]
    return wordi2ci, wordi2i, c_list, ci2wordi

if __name__ == '__main__':
    # test check_projective
    assert check_projective([0, 0, 2, 2]) == True
    assert check_projective([0, 0, 1, 3]) == False

    # test get dependency oracle
    cur_time = time.time()
    heads = [2, 0, 4, 5, 2, 2, 8, 6, 8, 9, 10, 13, 10, 13, 13, 8, 16, 17, 21, 21, 23, 23, 18] # should be single head trees
    print len(heads)
    print get_dep_oracle(heads)
    print time.time() - cur_time

else:
    pass
