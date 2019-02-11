from vi_syntax.vi.myio.Utils import check_projective, get_dep_oracle
from vi_syntax.vi.myio.Utils import read_cluster
from torchtext.data import Iterator, Batch
from torchtext import data
from torchtext import datasets
import sys


class VIDataset(object):

    def __init__(self):
        pass

    def build_wsj_enhanced_dataset(self, data_path=None, train_fname=None,
                                   valid_fname=None, test_fname=None, categorical_unk=True,
                                   filter_non_proj=True, min_length=0, max_length=9999,
                                   min_freq=2, vectors=None, vectors_cache=None, cluster_fname=None):
        """
        for generating enhanced dataset
        """
        if data_path is None or train_fname is None or valid_fname is None or test_fname is None:
            raise ValueError('missing data path/filename')

        intify = lambda x, *args: [int(token) for token in x]

        # Define the fields associated with the sequences.
        self.ID = data.Field(use_vocab=False, batch_first=True,
                             postprocessing = data.Pipeline(intify),
                             pad_token='-1')
        self.WORD = data.CategoricalUnkField(batch_first=True, include_lengths=True) \
            if categorical_unk else data.Field(batch_first=True, include_lengths=True)
        self.POS_TAG = data.Field(batch_first=True)
        self.DEP_HEAD = data.Field(use_vocab=False, batch_first=True,
                                   postprocessing = data.Pipeline(intify),
                                   pad_token='-1')
        self.CLUSTER_IDX = data.Field(use_vocab=False, batch_first=True,
                                      postprocessing = data.Pipeline(intify),
                                      pad_token='-1')
        self.INTRA_CLUSTER_IDX = data.Field(use_vocab=False, batch_first=True,
                                            postprocessing = data.Pipeline(intify),
                                            pad_token='-1')
        self.ACTION = data.Field(use_vocab=False, batch_first=True,
                                 postprocessing = data.Pipeline(intify),
                                 pad_token='-1')

        wsj_fields = [('id', self.ID), ('word', self.WORD), (None, None),
                      (None, None), ('pos_tag', self.POS_TAG), (None, None),
                      ('dep_head', self.DEP_HEAD), (None, None), (None, None),
                      (None, None), ('cluster_idx', self.CLUSTER_IDX),
                      ('intra_cluster_idx', self.INTRA_CLUSTER_IDX)]

        def length_filter(x): return len(
            x.word) >= min_length and len(x.word) <= max_length

        def sample_filter(x):
            if not filter_non_proj:
                return length_filter(x)
            else:
                return length_filter(x) and check_projective(x.dep_head)

        self.train, self.valid, self.test = datasets.SequenceTaggingDataset.splits(path=data_path, train=train_fname,
                                                                                   validation=valid_fname, test=test_fname,
                                                                                   fields=wsj_fields, filter_pred=sample_filter)

        # add shift-reduce action field
        for subdataset in(self.train, self.valid, self.test):
            subdataset.fields['action'] = self.ACTION
            for i in range(len(subdataset)):
                setattr(subdataset.examples[i], 'action', get_dep_oracle(
                    subdataset.examples[i].dep_head))                

        print 'train: {:d}, valid: {:d}, test: {:d}'.format(
            len(self.train), len(self.valid), len(self.test))

        self.WORD.build_vocab(self.train.word, min_freq=min_freq, vectors=vectors, vectors_cache=vectors_cache)
        self.POS_TAG.build_vocab(self.train.pos_tag)

        if cluster_fname is not None:
            self.wordi2ci, self.wordi2i, _, self.ci2wordi = read_cluster(cluster_fname, self.WORD.vocab.stoi)

    def build_ud_enhanced_dataset(self, data_path=None, train_fname=None,
                                  valid_fname=None, test_fname=None,
                                  filter_non_proj=True, min_length=0, max_length=9999,
                                  min_freq=2, vectors=None, vectors_cache=None, cluster_fname=None):
        """
        for generating enhanced dataset
        """
        if data_path is None or train_fname is None or valid_fname is None or test_fname is None:
            raise ValueError('missing data path/filename')

        def intify(x, *args):
            return [int(token) for token in x]
        
        def mylower(x, *args):
            return x.lower()

        # Define the fields associated with the sequences.
        self.ID = data.Field(use_vocab=False, batch_first=True,
                             postprocessing = data.Pipeline(intify),
                             pad_token='-1')
        self.WORD = data.Field(batch_first=True, include_lengths=True, preprocessing = data.Pipeline(mylower))  # lower for UD
        self.POS_TAG = data.Field(batch_first=True)
        self.DEP_HEAD = data.Field(use_vocab=False, batch_first=True,
                                   postprocessing = data.Pipeline(intify),
                                   pad_token='-1')
        self.CLUSTER_IDX = data.Field(use_vocab=False, batch_first=True,
                                      postprocessing = data.Pipeline(intify),
                                      pad_token='-1')
        self.INTRA_CLUSTER_IDX = data.Field(use_vocab=False, batch_first=True,
                                            postprocessing = data.Pipeline(intify),
                                            pad_token='-1')
        self.ACTION = data.Field(use_vocab=False, batch_first=True,
                                 postprocessing = data.Pipeline(intify),
                                 pad_token='-1')

        ud_fields = [('id', self.ID), ('word', self.WORD), (None, None),
                     ('pos_tag', self.POS_TAG), (None, None), (None, None),
                     ('dep_head', self.DEP_HEAD), (None, None), (None, None),
                     (None, None), ('cluster_idx', self.CLUSTER_IDX),
                     ('intra_cluster_idx', self.INTRA_CLUSTER_IDX)]

        def length_filter(x): return len(
            x.word) >= min_length and len(x.word) <= max_length

        def sample_filter(x):
            if not filter_non_proj:
                return length_filter(x)
            else:
                return length_filter(x) and check_projective(x.dep_head)

        self.train, self.valid, self.test = datasets.SequenceTaggingDataset.splits(path=data_path, train=train_fname,
                                                                                   validation=valid_fname, test=test_fname,
                                                                                   fields=ud_fields, filter_pred=sample_filter)

        # add shift-reduce action field
        for subdataset in(self.train, self.valid, self.test):
            subdataset.fields['action'] = self.ACTION
            for i in range(len(subdataset)):
                setattr(subdataset.examples[i], 'action', get_dep_oracle(
                    subdataset.examples[i].dep_head))                

        print 'train: {:d}, valid: {:d}, test: {:d}'.format(
            len(self.train), len(self.valid), len(self.test))

        self.WORD.build_vocab(self.train.word, min_freq=min_freq, vectors=vectors, vectors_cache=vectors_cache)
        self.POS_TAG.build_vocab(self.train.pos_tag)

        if cluster_fname is not None:
            self.wordi2ci, self.wordi2i, _, self.ci2wordi = read_cluster(cluster_fname, self.WORD.vocab.stoi)

    def build_wsj_dataset(self, data_path=None, train_fname=None,
                          valid_fname=None, test_fname=None, categorical_unk=True,
                          filter_non_proj=True, min_length=0, max_length=9999,
                          min_freq=2):
        """
        for generating enhanced dataset
        """
        if data_path is None or train_fname is None or valid_fname is None or test_fname is None:
            raise ValueError('missing data path/filename')

        # Define the fields associated with the sequences.
        self.WORD = data.CategoricalUnkField() if categorical_unk else data.Field()
        self.CPOS_TAG = data.Field()
        self.POS_TAG = data.Field()
        self.DEP_HEAD = data.Field(use_vocab=False)
        self.CLUSTER_IDX = data.Field(use_vocab=False)
        self.INTRA_CLUSTER_IDX = data.Field(use_vocab=False)

        wsj_fields = [(None, None), ('word', self.WORD), (None, None),
                      ('cpos_tag', self.CPOS_TAG), ('pos_tag', self.POS_TAG), (None, None),
                      ('dep_head', self.DEP_HEAD)]
    
        def length_filter(x): return len(
            x.word) >= min_length and len(x.word) <= max_length

        def sample_filter(x):
            if not filter_non_proj:
                return length_filter(x)
            else:
                return length_filter(x) and check_projective(x.dep_head)

        self.train, self.valid, self.test = datasets.SequenceTaggingDataset.splits(path=data_path, train=train_fname,
                                                                    validation=valid_fname, test=test_fname,
                                                                    fields=wsj_fields, filter_pred=sample_filter)
        print 'train: {:d}, valid: {:d}, test: {:d}'.format(len(self.train), len(self.valid), len(self.test))

        self.WORD.build_vocab(self.train.word, min_freq=min_freq)
        self.POS_TAG.build_vocab(self.train.pos_tag)
        print 'vocab: {:d}'.format(len(self.WORD.vocab))

    def build_ud_dataset(self, data_path=None, train_fname=None,
                         valid_fname=None, test_fname=None, categorical_unk=True,
                         filter_non_proj=True, min_length=0, max_length=9999,
                         min_freq=2):
        """
        for generating enhanced dataset
        """
        if data_path is None or train_fname is None or valid_fname is None or test_fname is None:
            raise ValueError('missing data path/filename')

        def mylower(x, *args):
            return x.lower()

        # Define the fields associated with the sequences.
        self.ID = data.Field(use_vocab=False)
        self.WORD = data.Field(preprocessing=data.Pipeline(mylower))
        self.CPOS_TAG = data.Field()
        self.POS_TAG = data.Field()
        self.DEP_HEAD = data.Field(use_vocab=False)
        self.DEP_REL = data.Field()

        ud_fields = [('id', self.ID), ('word', self.WORD), (None, None),
                      ('cpos_tag', self.CPOS_TAG), ('pos_tag',self.POS_TAG), (None, None),
                      ('dep_head', self.DEP_HEAD), ('dep_rel',self.DEP_REL), (None, None),
                      (None, None)]

        def length_filter(x): return len(
            x.word) >= min_length and len(x.word) <= max_length

        def sample_filter(x):
            if not filter_non_proj:
                return length_filter(x)
            else:
                return length_filter(x) and check_projective(x.dep_head)
            
        self.train, self.valid, self.test = datasets.SequenceTaggingDataset.splits(path=data_path, train=train_fname,
                                                                    validation=valid_fname, test=test_fname,
                                                                    fields=ud_fields, filter_pred=sample_filter)


        self.WORD.build_vocab(self.train.word, min_freq=min_freq)
        self.CPOS_TAG.build_vocab(self.train.cpos_tag)
        self.POS_TAG.build_vocab(self.train.pos_tag)
