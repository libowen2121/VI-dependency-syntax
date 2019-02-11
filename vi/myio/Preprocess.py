"""
    preprocess wsj and ud treebank and add fields:
            shift-reduce action, cluster index, intra cluster index
"""
import sys
from IO import VIDataset
from Utils import read_cluster
import os.path

def preprocess_wsj(data_path, train_fname, valid_fname,
                   test_fname, target_path, train_enhanced_fname,
                   valid_enhanced_fname, test_enhanced_fname, cluster_fname=None):
    dataset = VIDataset()
    dataset.build_wsj_dataset(data_path=data_path, train_fname=train_fname, valid_fname=valid_fname,
                              test_fname=test_fname)

    if cluster_fname is not None:
        wordi2ci, wordi2i, _, _ = read_cluster(cluster_fname, dataset.WORD.vocab.stoi)
        
    for fname, subdataset in zip([train_enhanced_fname, valid_enhanced_fname, test_enhanced_fname], 
                                 [dataset.train, dataset.valid, dataset.test]):
        with open(os.path.join(target_path, fname), 'w') as f:
            for i in range(len(subdataset.examples)):
                example = subdataset.examples[i]
                cluster_indices = [wordi2ci[dataset.WORD.vocab.stoi[token]] for token in example.word]
                intra_cluster_indices = [
                    wordi2i[dataset.WORD.vocab.stoi[token]] for token in example.word]
                lines = get_conll_sample_str(range(1, len(example.word)+1), example.word,
                                             ['_'] * len(example.word), ['_'] * len(example.word),
                                             example.pos_tag, ['_'] * len(example.word),
                                             example.dep_head, ['_'] * len(example.word),
                                             ['_'] * len(example.word), ['_'] * len(example.word),
                                             cluster_indices, intra_cluster_indices)
                f.write(lines)
                f.write('\n')


def preprocess_ud(data_path, train_fname, valid_fname,
                  test_fname, target_path, train_enhanced_fname,
                  valid_enhanced_fname, test_enhanced_fname, cluster_fname=None):
    dataset = VIDataset()
    dataset.build_ud_dataset(data_path=data_path, train_fname=train_fname, valid_fname=valid_fname,
                             test_fname=test_fname)

    if cluster_fname is not None:
        wordi2ci, wordi2i, _, _ = read_cluster(cluster_fname, dataset.WORD.vocab.stoi)

    for fname, subdataset in zip([train_enhanced_fname, valid_enhanced_fname, test_enhanced_fname],
                                 [dataset.train, dataset.valid, dataset.test]):
        with open(os.path.join(target_path, fname), 'w') as f:
            for i in range(len(subdataset.examples)):
                example = subdataset.examples[i]
                cluster_indices = [wordi2ci[dataset.WORD.vocab.stoi[token]] for token in example.word]
                intra_cluster_indices = [wordi2i[dataset.WORD.vocab.stoi[token]] for token in example.word]
                lines = get_conll_sample_str(range(1, len(example.word)+1), example.word,
                                             ['_'] * len(example.word), example.cpos_tag,
                                             example.pos_tag, ['_'] * len(example.word),
                                             example.dep_head, example.dep_rel,
                                             ['_'] * len(example.word), ['_'] * len(example.word),
                                             cluster_indices, intra_cluster_indices)
                f.write(lines)
                f.write('\n')
    return dataset

def get_conll_sample_str(*args):
    """
    Arguments:
        fp: file pointer
        *args: lists
    """
    column = len(args)
    lines = ''
    for x in zip(*args):
        for i in range(column):
            lines += str(x[i]) if not isinstance(x[i], str) else x[i]
            lines += '\t'
        lines += '\n'
    return lines


if __name__ == '__main__':
    # # generate enhanced wsj conll dataset
    # data_path = '/Users/boon/code/study/corpora/wsj dependency/'
    # train_fname = 'wsj-inf_2-21_dep'
    # valid_fname = 'wsj-inf_22_dep'
    # test_fname = 'wsj-inf_23_dep'
    # target_path = '/Users/boon/Dropbox/code/study/pytorch/vi_syntax/data/wsj'
    # train_enhanced_fname = 'wsj_train_enhanced'
    # valid_enhanced_fname = 'wsj_valid_enhanced'
    # test_enhanced_fname = 'wsj_test_enhanced'
    # cluster_fname = '../../data/cluster/clusters-train-berk.txt'
    # preprocess_wsj(data_path, train_fname, valid_fname,
    #                test_fname, target_path, train_enhanced_fname,
    #                valid_enhanced_fname, test_enhanced_fname, cluster_fname)

    # generate enhanced ud conll dataset
    language_id = 'pt'
    data_path = '/Users/boon/Dropbox/code/study/pytorch/vi_syntax/data/ud'
    train_fname = language_id + '-ud-train_clean.conllu'
    valid_fname = language_id + '-ud-dev_clean.conllu'
    test_fname = language_id + '-ud-test_clean.conllu'
    target_path = '/Users/boon/Dropbox/code/study/pytorch/vi_syntax/data/ud'
    train_enhanced_fname = language_id + '_train_enhanced'
    valid_enhanced_fname = language_id + '_valid_enhanced'
    test_enhanced_fname = language_id + '_test_enhanced'
    cluster_fname = '../../data/cluster/' + language_id + '_cluster'
    dataset = preprocess_ud(data_path, train_fname, valid_fname,
                  test_fname, target_path, train_enhanced_fname,
                  valid_enhanced_fname, test_enhanced_fname, cluster_fname)
    pass
else:
    pass
