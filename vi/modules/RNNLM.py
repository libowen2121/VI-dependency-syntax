import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.autograd import Variable

import numpy as np

class LanguageModel(nn.Module):

    def __init__(self, 
                 word_dim, 
                 pos_dim, 
                 lstm_dim, 
                 nlayers, 
                 dropout,
                 batchsize=1,
                 tie_weights=False, 
                 pretrain=False,
                 pretrain_word_vectors=None,                 
                 w2i=None,
                 i2w=None,
                 w2i_pos=None,
                 i2w_pos=None,
                 para_init=None,
                 init_name=None,
                 gpu_id=-1):
        super(LanguageModel, self).__init__()
        self.word_dim = word_dim
        self.pos_dim = pos_dim
        self.lstm_dim = lstm_dim
        self.nlayers = nlayers
        self.dropout = dropout
        self.batchsize = batchsize
        self.tie_weights = tie_weights
        self.pretrain = pretrain
        self.i2w = i2w
        self.w2i = w2i
        self.i2w_pos = i2w_pos
        self.w2i_pos = w2i_pos
        self.len_word = len(self.i2w)
        self.len_pos = len(self.i2w_pos)
        self.para_init = para_init
        self.init_name = init_name
        self.gpu_id = gpu_id

        assert self.word_dim * self.pos_dim == 0    # only one feature

        # build nn blocks
        if self.w2i['<pad>'] != self.w2i_pos['<pad>']:
            raise ValueError('index of <pad> is not consisent')
        self.pad_idx = self.w2i['<pad>']
        if self.word_dim > 0:
            self.embedding = nn.Embedding(self.len_word, self.word_dim, padding_idx=self.pad_idx)
            self.decoder = nn.Linear(self.lstm_dim, self.len_word)
            self.input_dim = self.word_dim
        if self.pos_dim > 0:
            self.embedding = nn.Embedding(self.len_pos, self.pos_dim, padding_idx=self.pad_idx)
            self.decoder = nn.Linear(self.lstm_dim, self.len_pos)
            self.input_dim = self.pos_dim
        self.criterion = nn.CrossEntropyLoss(size_average=False, ignore_index=self.pad_idx)
        if self.dropout > 0:
            self.dp = nn.Dropout(p=self.dropout)
        self.lm_rnn = nn.LSTM(input_size=self.input_dim, hidden_size=self.lstm_dim,
                              num_layers=nlayers, dropout=dropout)

        # initialize
        if self.para_init is not None:
            if self.init_name == 'glorot':
                for name, para in self.named_parameters():
                    print 'initializing', name
                    if len(para.size()) < 2:
                        para.data.zero_()
                    else:
                        self.para_init(para)
            else:
                for name, para in self.named_parameters():
                    print 'initializing', name
                    self.para_init(para)

        # load pretrain word embeddings
        if self.pretrain:
            # only for word embeddings
            self.load_embeddings(pretrain_word_vectors)

        # tie weights
        if self.tie_weights:
            if self.input_dim != self.lstm_dim:
                raise ValueError('When using the tied flag, input dim must be equal to lstm hidden dim')
            self.decoder.weight = self.embedding.weight
            if self.pretrain:
                for para in self.decoder.parameters():
                    para.requires_grad = False

        self.hidden0 = self.init_hidden()
        self.empty_emb = torch.randn(1, self.batchsize, self.input_dim) # for <s>
        if self.gpu_id > -1:
            self.empty_emb = self.empty_emb.cuda()
        self.empty_emb = Variable(self.empty_emb)

        if gpu_id > -1:
            self.cuda()

    def init_hidden(self):
        if self.gpu_id > -1:
            return (Variable(torch.zeros(self.nlayers, self.batchsize, self.input_dim).cuda()),
                    Variable(torch.zeros(self.nlayers, self.batchsize, self.input_dim).cuda()))
        else:
            return (Variable(torch.zeros(self.nlayers, self.batchsize, self.input_dim)),
                    Variable(torch.zeros(self.nlayers, self.batchsize, self.input_dim)))

    def load_embeddings(self, pretrain_word_vectors):
        self.embedding.weight.data.copy_(pretrain_word_vectors)
        for para in self.embedding.parameters():
            para.requires_grad = False

    def save(self, filename):
        torch.save(self.state_dict(), filename)

    def forward(self, words, pos_tags):
        """
        language model with a batch of variable lengths
        Arguments:
            words(Variable): words and lengths variables
            pos_tags(Variable):
            oracle_actions(Variable):
        Returns:
            loss(Variable): - log prob   
        """
        temp = self.batchsize

        if self.word_dim > 0:
            tokens, seq_lengths = words
        else:
            _, seq_lengths = words
            tokens = pos_tags
        seq_size = tokens.size(1)
        self.batchsize = tokens.size(0)

        input_seq = tokens.clone()
        target = tokens.clone()        
        # remove the last token
        if input_seq.size(1) == 1:
            input_seq = self.empty_emb
        else:
            for i in range(self.batchsize):
                input_seq[i, seq_lengths[i] - 1] = self.pad_idx
            input_seq = input_seq[:, : - 1]
            input_seq = self.embedding(input_seq)
            input_seq = torch.transpose(input_seq, 0, 1).contiguous()   # seq_len x batchsize x embed_dim
            input_seq = torch.cat((self.empty_emb[:, :self.batchsize, :], input_seq), 0)
        input_seq = self.dp(input_seq)

        packed_input = pack_padded_sequence(input_seq, seq_lengths.cpu().numpy())
        packed_output, _ = self.lm_rnn(packed_input,
                                     (self.hidden0[0][:, :self.batchsize, :].contiguous(),
                                      self.hidden0[1][:, :self.batchsize, :].contiguous()))
        output, _ = pad_packed_sequence(packed_output)
        output = self.dp(output)    # seq_size x batchsize x lstm_dim
        output = output.view(seq_size * self.batchsize, self.lstm_dim)
        output = self.decoder(output)   # seq_size * batchsize x len_token
        output = output.view(seq_size, self.batchsize, -1)
        output = torch.transpose(output, 0, 1).contiguous() # batch first
        loss = self.criterion(output.view(self.batchsize * seq_size, -1), target.view(-1))   

        self.batchsize = temp

        return loss

    # def generate(self, length, temperature):
    #     sent = ''
    #     input = self.empty_emb
    #     hidden = self.hidden0
    #     for i in range(length):
    #         output, hidden = self.lm_rnn(input, hidden)
    #         output_weights = output.squeeze().data.div(temperature).exp()
    #         token_idx = torch.multinomial(output_weights, 1)[0]
    #         if self.word_dim > 0:
    #             token = self.i2w[token_idx]
    #         else:
    #             token = self.i2w_pos[token_idx]
    #         sent += token
    #         sent += ' '
    #         if self.gpu_id > -1:
    #             token_idx = Variable(torch.LongTensor([[token_idx]]).cuda())
    #         else:
    #             token_idx = Variable(torch.LongTensor([[token_idx]]))
    #         input = self.embedding(token_idx)
    #     return sent

class Baseline_linear(nn.Module):
    """
    add a 1 to 1 linear layer to language model baseline
    """
    def __init__(self, gpu_id=-1):
        super(Baseline_linear, self).__init__()
        self.a = nn.Linear(1, 1)
        if gpu_id > -1:
            self.a.cuda()
        init.constant_(self.a.weight, 1.)
        init.constant_(self.a.bias, 0.)
        
    def forward(self, x):
        return self.a(x)




