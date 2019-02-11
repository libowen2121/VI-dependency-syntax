import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from vi_syntax.vi.myio.Utils import SHIFT, REDUCE_L, REDUCE_R

class Decoder(nn.Module):

    def __init__(self,
                 nlayers,
                 pos_dim,
                 action_dim,
                 lstm_dim,
                 dropout,
                 word_dim,
                 pretrain_word_dim,
                 pretrain_word_vectors=None,
                 w2i=None,
                 i2w=None,
                 w2i_pos=None,
                 i2w_pos=None,
                #  H1=False,
                #  H1_decay=5.,
                #  H2=False,
                #  H2_decay=5.,
                 cluster=True,
                 wi2ci=None,
                 wi2i=None,
                 ci2wi=None,
                 para_init=None,
                 init_name=None,
                 gpu_id=-1,
                 seed=-1):

        super(Decoder, self).__init__()

        if seed > 0:
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
        self.i2w = i2w
        self.w2i = w2i
        self.i2w_pos = i2w_pos
        self.w2i_pos = w2i_pos
        self.pos_dim = pos_dim
        self.word_dim = word_dim
        self.pretrain_word_dim = pretrain_word_dim
        self.nlayers = nlayers
        self.action_dim = action_dim
        self.len_word = len(self.i2w)
        self.len_pos = len(self.i2w_pos)
        self.lstm_dim = lstm_dim
        self.len_act = 3
        self.parser_state_dim = 3 * self.lstm_dim if self.action_dim > 0 else 2 * self.lstm_dim
        self.dropout = dropout
        self.cluster = cluster
        self.para_init = para_init
        self.init_name = init_name
        self.gpu_id = gpu_id

        self.SHIFT = SHIFT      # GEN for decoder
        self.REDUCE_L = REDUCE_L
        self.REDUCE_R = REDUCE_R

        # nn blocks
        self.input_proj = nn.Linear(self.word_dim + self.pos_dim, self.lstm_dim, False)
        self.comp = nn.Linear(2 * self.lstm_dim, self.lstm_dim)
        self.mlp = nn.Linear(self.parser_state_dim, self.lstm_dim)
        self.act_output = nn.Linear(self.lstm_dim, self.len_act)

        if self.dropout > 0:
            self.dp = nn.Dropout(p=self.dropout)

        self.buffer_rnn = nn.LSTM(input_size=self.lstm_dim, hidden_size=self.lstm_dim, num_layers=nlayers)
        self.stack_rnn = nn.LSTM(input_size=self.lstm_dim, hidden_size=self.lstm_dim, num_layers=nlayers)
        self.stack_initial_state = self.get_initial_state(num_layers=self.nlayers, direction=1,
                                                          hidden_size=self.lstm_dim, batch_size=1)

        if self.word_dim > 0:
            self.word_proj = nn.Linear(self.pretrain_word_dim, self.word_dim)
            self.word_embedding = nn.Embedding(self.len_word, self.pretrain_word_dim)
            if not self.cluster:
                self.word_output = nn.Linear(self.lstm_dim, self.len_word)
            else:
                self.wordi2i = wi2i
                self.wordi2ci = wi2ci
                self.ci2wordi = ci2wi
                self.len_cluster = len(self.ci2wordi)
                self.cluster_output = nn.Linear(self.lstm_dim, self.len_cluster)
                self.word_output_l = []
                for c in self.ci2wordi:
                    if len(self.ci2wordi[c]) < 1:
                        self.word_output_l.append(None)
                    else:
                        self.word_output_l.append(nn.Linear(self.lstm_dim, len(self.ci2wordi[c])))
                        if self.gpu_id > -1:
                            self.word_output_l[-1].cuda()

        if self.pos_dim > 0:
            self.pos_output = nn.Linear(self.lstm_dim, self.len_pos)
            self.pos_embedding = nn.Embedding(self.len_pos, pos_dim)
        if self.action_dim > 0:
            self.act_rnn = nn.LSTM(input_size=self.lstm_dim, hidden_size=self.lstm_dim, num_layers=nlayers)
            self.act_proj = nn.Linear(action_dim, self.lstm_dim)
            self.act_embedding = nn.Embedding(self.len_act, action_dim)
            self.act_initial_state = self.get_initial_state(num_layers=self.nlayers, direction=1,
                                                            hidden_size=self.lstm_dim, batch_size=1)

        self.empty_buffer_emb = torch.randn(1, self.lstm_dim)
        self.empty_buffer_emb = nn.Parameter(self.empty_buffer_emb, requires_grad=False)
        self.empty_stack_emb = torch.randn(1, self.lstm_dim)
        self.empty_stack_emb = nn.Parameter(self.empty_stack_emb, requires_grad=False)
        if self.action_dim > 0:
            self.empty_act_emb = torch.randn(1, self.lstm_dim)
            self.empty_act_emb = nn.Parameter(self.empty_buffer_emb, requires_grad=False)

        if self.para_init is not None:
            if self.init_name == 'glorot':
                for name, para in self.named_parameters():
                    # print 'initializing', name
                    if len(para.size()) < 2:
                        para.data.zero_()
                    else:
                        self.para_init(para)
            else:
                for name, para in self.named_parameters():
                    # print 'initializing', name
                    self.para_init(para)

        if self.word_dim > 0 and self.pretrain_word_dim > 0:
            self.load_embeddings(pretrain_word_vectors)

        if self.gpu_id > -1:
            self.cuda()

    def load_embeddings(self, pretrain_word_vectors):
        self.word_embedding.weight.data.copy_(pretrain_word_vectors)

    def save(self, filename):
        torch.save(self.state_dict(), filename)

    def get_initial_state(self, num_layers, direction, hidden_size, batch_size):
        '''
            initial states for LSTMs
        '''
        if self.gpu_id > -1:
            h0_tensor = torch.zeros(num_layers * direction, batch_size, hidden_size).cuda()
            c0_tensor = torch.zeros(num_layers * direction, batch_size, hidden_size).cuda()
        else:
            h0_tensor = torch.zeros(num_layers * direction, batch_size, hidden_size)
            c0_tensor = torch.zeros(num_layers * direction, batch_size, hidden_size)
        return (Variable(h0_tensor), Variable(c0_tensor))

    def encode_sentence(self, words, pos_tags):
        '''
            To score a sentence or parse tree, we can encode the forward tokens all at once.
            This will not be used in generation mode.
            Arguments:
                words(Variable):
                pos_tags(Variable):
        '''
        buffer_initial_state = self.get_initial_state(self.nlayers, 1, self.lstm_dim, 1)
        input_sent = None   # input embeddings
        len_tokens = words.size(1)

        if self.word_dim > 0:
            input_word = self.word_proj(self.word_embedding(words).view(-1, self.pretrain_word_dim)) \
                .view(len_tokens, 1, self.word_dim)  # length x 1 x word_dim
            input_word = F.relu(input_word)
            input_sent = input_word

        if self.pos_dim > 0:
            input_pos = self.pos_embedding(pos_tags).view(len_tokens, 1, self.pos_dim)  # length x 1 x pos_dim
            if input_sent is not None:
                input_sent = torch.cat((input_sent, input_pos), 2)  # length x 1 x dim
            else:
                input_sent = input_pos

        input_sent = F.relu(self.input_proj(input_sent.view(len_tokens, self.word_dim + self.pos_dim))) \
            .view(len_tokens, 1, self.lstm_dim)  # len_tokens x 1 x self.lstm_dim

        buffer_sent, _ = self.buffer_rnn(input_sent, buffer_initial_state)  # len_tokens x 1 x self.lstm_dim

        input_sent = [input_sent[idx] for idx in range(len_tokens)]      # 1 x self.lstm_dim
        buffer_sent = [buffer_sent[idx] for idx in range(len_tokens)]    # 1 x self.lstm_dim

        return input_sent, buffer_sent

    def forward(self, words, pos_tags, oracle_actions):
        '''
            compute loss of the given actions
            Arguments:
                words(Variable):
                pos_tags(Variable):
                oracle_actions(Variable):
        '''
        if isinstance(oracle_actions, list):
            if self.gpu_id > -1:
                oracle_actions = Variable(torch.LongTensor(oracle_actions).unsqueeze(0).cuda())
            else:
                oracle_actions = Variable(torch.LongTensor(oracle_actions).unsqueeze(0))
        oracle_actions = [oracle_actions[0, i] for i in range(oracle_actions.size(1))] # copy list (Variable)

        tokens = [(words.data[0,i], pos_tags.data[0,i])  for i in range(words.size(1))]

        stack = []
        if self.action_dim > 0:
            act_state = self.act_initial_state      # state of act rnn
            act_summary = self.empty_act_emb  # output of act rnn

        input_sent, buffer_sent = self.encode_sentence(words, pos_tags) # token_embeddings and buffer in original order
        loss_act = None
        loss_token = None
        input_sent.reverse()    # in reverse order
        buffer_sent.reverse()   # in reverse order
        tokens.reverse()        # in reverse order

        buffer_embedding = self.empty_buffer_emb

        while not (len(stack) == 1 and len(buffer_sent) == 0):
            valid_actions = []

            if len(buffer_sent) > 0:
                valid_actions += [self.SHIFT]
            if len(stack) >= 2:
                valid_actions += [self.REDUCE_L, self.REDUCE_R]

            action = oracle_actions.pop(0)  # Variable

            parser_state_l = []
            stack_embedding = self.empty_stack_emb if len(stack) == 0 else stack[-1][0]
            parser_state_l.append(stack_embedding.view(1, self.lstm_dim))
            parser_state_l.append(buffer_embedding)
            if self.action_dim > 0:
                act_summary = act_summary.view(1, self.lstm_dim)
                parser_state_l.append(act_summary)
            parser_state = torch.cat(parser_state_l, 1)
            h = F.relu(self.mlp(parser_state))   # 1 x self.lstm_dim
            if self.dropout > 0:
                h = self.dp(h)

            if len(valid_actions) > 1:
                log_probs = F.log_softmax(self.act_output(h), dim=1) # 1 x out_dim
                cur_loss_act = - log_probs[0, action.data.item()]
                if loss_act is None:
                    loss_act = cur_loss_act
                else:
                    loss_act += cur_loss_act

            if self.action_dim > 0:
                act_embedding = self.act_proj(self.act_embedding(action).view(1, self.action_dim)).view(1, 1, self.action_dim)     # 1 x 1 x dim
                act_summary, act_state = self.act_rnn(act_embedding, act_state)

            # execute the action to update the parser state
            if action.data.item() == self.SHIFT:

                token = tokens.pop()
                if self.pos_dim > 0:
                    log_probs_pos = F.log_softmax(self.pos_output(h), dim=1)
                    cur_loss_pos = - log_probs_pos[0, token[1]]
                    if loss_token is None:
                        loss_token = cur_loss_pos
                    else:
                        loss_token += cur_loss_pos
                if self.word_dim > 0:
                    cur_word_idx = token[0].item()
                    if not self.cluster:
                        log_probs_word = F.log_softmax(self.word_output(h), dim=1)
                        cur_loss_word = - log_probs_word[0, cur_word_idx]
                        if loss_token is None:
                            loss_token = cur_loss_word
                        else:
                            loss_token += cur_loss_word
                    else:
                        cur_word_intra_idx = self.wordi2i[cur_word_idx]
                        cur_c = self.wordi2ci[cur_word_idx]  # cluster idx
                        log_probs_cluster = F.log_softmax(self.cluster_output(h), dim=1)
                        log_probs_word = F.log_softmax(self.word_output_l[cur_c](h), dim=1) # given c
                        cur_loss_cluster = - log_probs_cluster[0, cur_c]
                        cur_loss_intra_cluster = - log_probs_word[0, cur_word_intra_idx]
                        if loss_token is None:
                            loss_token = cur_loss_cluster + cur_loss_intra_cluster
                        else:
                            loss_token += (cur_loss_cluster + cur_loss_intra_cluster)

                buffer_embedding = buffer_sent.pop()    # 1 x self.lstm_dim
                token_embedding = input_sent.pop()
                stack_state = stack[-1][1] if stack else self.stack_initial_state
                output, stack_state = self.stack_rnn(token_embedding.view(1, 1, self.lstm_dim), stack_state)
                stack.append((output, stack_state, token))

            else:
                right = stack.pop()
                left = stack.pop()
                head, modifier = (left, right) if action.data.item() == self.REDUCE_R else (right, left)
                top_stack_state = stack[-1][1] if len(stack) > 0 else self.stack_initial_state
                head_rep, head_tok = head[0], head[2]
                mod_rep, mod_tok = modifier[0], modifier[2]
                composed_rep = F.relu(self.comp(torch.cat([head_rep, mod_rep], 2).view(1, 2 * self.lstm_dim)))
                output, top_stack_state = self.stack_rnn(composed_rep.view(1, 1, self.lstm_dim), top_stack_state)
                stack.append((output, top_stack_state, head_tok))

        loss = loss_token if loss_act is None else loss_token + loss_act
        return loss, loss_act, loss_token
