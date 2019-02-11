import sys
from vi_syntax.vi.myio.Utils import SHIFT, REDUCE_L, REDUCE_R

def compute_dda(oracle_heads, act_seq, output=False, words=None, pos_tags=None, logger=None):
    '''
        compute directed dependency accuracy (correct and total)
        Arguments:
            oracle_heads(list): length = n
            act_seq(list): length = 2n - 1
            output(bool): whether to output tree
            words([str]):
            pos_tags([str]):
            logger():  
        Return:
            correct(int): number of correct heads
            n(int): number of total heads
    '''
    act_seq = act_seq[:]
    n = len(oracle_heads)
    assert len(act_seq) == 2 * n - 1

    stack = []
    buffer = range(1, n + 1)
    buffer.reverse()
    heads = [-1] * n

    while not (len(stack) == 1 and len(buffer) == 0 ):
        action = act_seq.pop(0)
        # execute the action
        if action == SHIFT:
            stack.append(buffer.pop())
        else:
            right = stack.pop()
            left = stack.pop()
            head, modifier = (left, right) if action == REDUCE_R else (right, left)
            stack.append(head)
            heads[modifier - 1] = head

    head = stack.pop()
    heads[head - 1] = 0     # head is ROOT

    correct = sum([1 if x == y else 0 for x,y in zip(oracle_heads, heads)])

    # output the parsing trees
    if output:
        logger.info(' '.join(words))
        logger.info('oracle')
        for i in range(n):
            mod_pos = pos_tags[i]
            head_idx = oracle_heads[i]
            if head_idx == 0:
                head_pos = 'ROOT'
            else:
                head_pos = pos_tags[head_idx - 1]
            logger.info('{:d} {} -> {:d} {}'.format(head_idx, head_pos, i + 1, mod_pos))
        logger.info('induced')
        for i in range(n):
            mod_pos = pos_tags[i]
            head_idx = heads[i]
            if head_idx == 0:
                head_pos = 'ROOT'
            else:
                head_pos = pos_tags[head_idx - 1]
            logger.info('{:d} {} -> {:d} {}'.format(head_idx, head_pos, i + 1, mod_pos))
        logger.info('correct: {:d}/{:d}'.format(correct, n))
        logger.info('')

    return correct, n


def compute_dda_long_dep(oracle_heads,
                act_seq,
                output=False,
                words=None,
                pos_tags=None,
                logger=None):
    '''
        compute directed dependency accuracy (correct and total) especially for long-distance dependency
        Arguments:
            oracle_heads(list): length = n
            act_seq(list): length = 2n - 1
            output(bool): whether to output tree
            words([str]):
            pos_tags([str]):
            logger():  
        Return:
            correct(int): number of correct heads
            n(int): number of total heads
    '''
    act_seq = act_seq[:]
    n = len(oracle_heads)
    assert len(act_seq) == 2 * n - 1

    stack = []
    buffer = range(1, n + 1)
    buffer.reverse()
    heads = [-1] * n

    while not (len(stack) == 1 and len(buffer) == 0):
        action = act_seq.pop(0)
        # execute the action
        if action == SHIFT:
            stack.append(buffer.pop())
        else:
            right = stack.pop()
            left = stack.pop()
            head, modifier = (left, right) if action == REDUCE_R else (right,
                                                                       left)
            stack.append(head)
            heads[modifier - 1] = head

    head = stack.pop()
    heads[head - 1] = 0  # head is ROOT

    correct = 0
    n = 0

    for i in range(len(oracle_heads)):
        if abs(oracle_heads[i] - i) >= 7:
            n += 1
            if oracle_heads[i] == heads[i]:
                correct += 1


    return correct, n

# rule translation for English WSJ
ROOT_SET = ['ROOT']
VERB_SET = ['VB','VBD','VBG','VBN','VBP', 'VBZ']
AUX_SET = ['MD']
ADV_SET = ['RB', 'RBR', 'RBS', 'WRB']
NOUN_SET = ['NN', 'NNS', 'NNP', 'NNPS']
PRON_SET = ['PRP', 'PRP$', 'WP', 'WP$']
ADJ_SET = ['JJ', 'JJR', 'JJS']
ART_SET = ['DT', 'PDT', 'WDT']
NUM_SET = ['CD']
PREP_SET = ['IN']


def get_rule_idx(pos_l, pos_r):
    if pos_l in ROOT_SET:
        if pos_r in AUX_SET:
            return 0
    if pos_l in ROOT_SET:
        if pos_r in VERB_SET:
            return 1
    if pos_l in VERB_SET:
        if pos_r in NOUN_SET:
            return 2
    if pos_l in VERB_SET:
        if pos_r in PRON_SET:
            return 3
    if pos_l in VERB_SET:
        if pos_r in ADV_SET:
            return 4
    if pos_l in VERB_SET:
        if pos_r in VERB_SET:
            return 5
    if pos_l in AUX_SET:
        if pos_r in VERB_SET:
            return 6
    if pos_l in NOUN_SET:
        if pos_r in ADJ_SET:
            return 7
    if pos_l in NOUN_SET:
        if pos_r in ART_SET:
            return 8
    if pos_l in NOUN_SET:
        if pos_r in NOUN_SET:
            return 9
    if pos_l in NOUN_SET:
        if pos_r in NUM_SET:
            return 10
    if pos_l in PREP_SET:
        if pos_r in NOUN_SET:
            return 11
    if pos_l in ADJ_SET:
        if pos_r in ADV_SET:
            return 12
    return -1

def compute_rule_acc(oracle_arcs, act_seq, tags):
    '''
        compute directed dependency accuracy (correct and total)
    '''
    oracle_arcs = [(x if isinstance(x, int) else x.item(), y) for x,y in oracle_arcs]

    act_seq = act_seq[:]
    n = (len(act_seq) + 1) / 2

    stack = []
    buffer = range(1, n + 1)
    buffer.reverse()
    arcs = set()

    while not (len(stack) == 1 and len(buffer) == 0 ):
        action = act_seq.pop(0)
        # execute the action
        if action == SHIFT:
            stack.append(buffer.pop())
        else:
            right = stack.pop()
            left = stack.pop()
            head, modifier = (left, right) if action == REDUCE_R else (right, left)
            stack.append(head)
            arcs.add((head, modifier))

    head = stack.pop()
    arcs.add((0, head))

    total_rule = [0.] * 13
    correct_rule = [0.] * 13

    assert len(arcs) == len(oracle_arcs)
    for arc in oracle_arcs:
        if arc[0] == 0:
            head_pos = 'ROOT'
        else:
            head_pos = tags[arc[0] - 1]
        mod_pos = tags[arc[1] - 1]
        rule_idx = get_rule_idx(head_pos, mod_pos)
        if rule_idx > -1:
            total_rule[rule_idx] += 1
            if arc in arcs:
                correct_rule[rule_idx] += 1

    return correct_rule, total_rule

if __name__ == '__main__':
    oracle_heads = [2, 0, 5, 5, 2]
    act_seq = [SHIFT, SHIFT, REDUCE_L, SHIFT, SHIFT, REDUCE_L, SHIFT, REDUCE_L, REDUCE_R]
    print compute_dda(oracle_heads, act_seq)
else:
    pass