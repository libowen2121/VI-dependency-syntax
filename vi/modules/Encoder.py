import torch
import dill
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from vi_syntax.vi.myio.Utils import REDUCE_L, REDUCE_R, SHIFT


class Encoder(nn.Module):
    '''
        encoder (inference network)
    '''

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
                 para_init=None,
                 init_name=None,
                 gpu_id=-1,
                 seed=-1):

        super(Encoder, self).__init__()

        if seed > 0:
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
        self.i2w = i2w
        self.w2i = w2i
        self.i2w_pos = i2w_pos
        self.w2i_pos = w2i_pos
        self.word_dim = word_dim
        self.pretrain_word_dim = pretrain_word_dim
        self.pos_dim = pos_dim
        self.nlayers = nlayers
        self.action_dim = action_dim
        self.len_word = len(self.i2w)
        self.len_pos = len(self.i2w_pos)
        self.len_act = 3
        self.dropout = dropout
        self.lstm_dim = lstm_dim
        self.parser_state_dim = 4 * self.lstm_dim if self.action_dim > 0 else 3 * self.lstm_dim
        self.para_init = para_init
        self.init_name = init_name
        self.gpu_id = gpu_id

        self.SHIFT = SHIFT
        self.REDUCE_L = REDUCE_L
        self.REDUCE_R = REDUCE_R

        # build nn blocks
        self.input_proj = nn.Linear(word_dim + pos_dim, self.lstm_dim, False)
        self.buffer_rnn = nn.LSTM(input_size=self.lstm_dim, hidden_size=self.lstm_dim,
                                  num_layers=nlayers, bidirectional=True)
        self.comp = nn.Linear(2 * self.lstm_dim, self.lstm_dim)
        self.stack_rnn = nn.LSTM(input_size=self.lstm_dim, hidden_size=self.lstm_dim,
                                 num_layers=nlayers)
        if dropout > 0:
            self.dp = nn.Dropout(p=self.dropout)
        self.mlp = nn.Linear(self.parser_state_dim, self.lstm_dim)
        self.act_output = nn.Linear(self.lstm_dim, self.len_act)

        if self.word_dim > 0:
            self.pretrain_proj = nn.Linear(pretrain_word_dim, word_dim)
            self.word_embedding = nn.Embedding(self.len_word, pretrain_word_dim)
        if self.action_dim > 0:
            self.act_rnn = nn.LSTM(input_size=self.lstm_dim, hidden_size=self.lstm_dim,
                                   num_layers=nlayers)
            self.act_proj = nn.Linear(action_dim, self.lstm_dim)
            self.act_embedding = nn.Embedding(self.len_act, action_dim)
            self.act_initial_state = self.get_initial_state(num_layers=self.nlayers, direction=1,
                                                            hidden_size=self.lstm_dim, batch_size=1)
        if self.pos_dim > 0:
            self.pos_embedding = nn.Embedding(self.len_pos, pos_dim)
        self.stack_initial_state = self.get_initial_state(num_layers=self.nlayers, direction=1,
                                                          hidden_size=self.lstm_dim, batch_size=1)

        self.empty_buffer_emb = torch.randn(1, 2 * self.lstm_dim)
        self.empty_buffer_emb = nn.Parameter(self.empty_buffer_emb, requires_grad=False)
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
        for para in self.word_embedding.parameters():
            para.requires_grad = False

    def save(self, filename):
        torch.save(self.state_dict(), filename)

    def get_initial_state(self, num_layers, direction, hidden_size, batch_size):
        """
            initial states for LSTMs
        """
        if self.gpu_id > -1:
            h0_tensor = torch.zeros(num_layers * direction, batch_size, hidden_size).cuda()
            c0_tensor = torch.zeros(num_layers * direction, batch_size, hidden_size).cuda()
        else:
            h0_tensor = torch.zeros(num_layers * direction, batch_size, hidden_size)
            c0_tensor = torch.zeros(num_layers * direction, batch_size, hidden_size)
        return (Variable(h0_tensor), Variable(c0_tensor))

    def encode_sentence(self, inv_words, inv_pos_tags):
        """
            sentence encoding for bi-directional buffer
            Arguments:
                inv_words([Variable]):
                inv_pos_tags([Variable]):
            Returns:
                input_sent([Variable]):
                buffer_sent([Variable]):
        """
        buffer_initial_state = self.get_initial_state(self.nlayers, 2, self.lstm_dim, 1)
        input_sent = None  # input embeddings
        len_tokens = inv_words.size(1)

        if self.word_dim > 0:
            input_word = self.pretrain_proj(self.word_embedding(inv_words).view(-1, self.pretrain_word_dim)) \
                                            .view(len_tokens, 1, self.word_dim)  # length x 1 x word_dim
            input_word = F.relu(input_word)
            input_sent = input_word
        if self.pos_dim > 0:
            input_pos = self.pos_embedding(inv_pos_tags).view(len_tokens, 1, self.pos_dim)  # length x 1 x pos_dim
            if input_sent is not None:
                input_sent = torch.cat((input_sent, input_pos), 2)  # length x 1 x dim
            else:
                input_sent = input_pos

        input_sent = F.relu(self.input_proj(input_sent.view(len_tokens, self.word_dim + self.pos_dim))) \
            .view(len_tokens, 1, self.lstm_dim)  # length x 1 x lstm_dim

        # len_tokens x 1 x 2*lstm_dim
        buffer_sent, _ = self.buffer_rnn(input_sent, buffer_initial_state)

        input_sent = [input_sent[idx] for idx in range(len_tokens)]  # 1 x lstm_dim
        buffer_sent = [buffer_sent[idx] for idx in range(len_tokens)]  # 1 x 2*lstm_dim

        return input_sent, buffer_sent

    def train_parser(self, words, pos_tags, oracle_actions):
        """
        train encoder
        Arguments:
            words(Variable):
            pos_tags(Variable):
            oracle_actions(Variable):
        Returns:
            loss_act(Variable):
        """

        oracle_actions = [oracle_actions[0, i] for i in range(oracle_actions.size(1))] # copy list
        # print [oracle_actions[i].data[0] for i in range(len(oracle_actions))]

        if self.gpu_id > -1:
            word_inv_idx = Variable(torch.arange(words.size(1) - 1, -1, -1).long().cuda())
            pos_inv_idx = Variable(torch.arange(pos_tags.size(1) - 1, -1, -1).long().cuda())
        else:
            word_inv_idx = Variable(torch.arange(words.size(1) - 1, -1, -1).long())
            pos_inv_idx = Variable(torch.arange(pos_tags.size(1) - 1, -1, -1).long())
        # get token_embeddings and buffer (in reversed order)
        input_sent, buffer_sent=self.encode_sentence(words.index_select(1, word_inv_idx),
                                                      pos_tags.index_select(1, pos_inv_idx))
        assert len(buffer_sent) * 2 - 1 == len(oracle_actions)
        stack = []  # stack LSTM
        if self.action_dim > 0:
            act_state = self.act_initial_state  # state of act rnn
            act_summary = self.empty_act_emb    # output of act rnn

        loss_act = None

        while not (len(stack) == 1 and len(buffer_sent) == 0):
            valid_actions = []  # based on parser state, get valid actions

            if len(buffer_sent) > 0:
                valid_actions += [self.SHIFT]
            if len(stack) >= 2:
                valid_actions += [self.REDUCE_L, self.REDUCE_R]

            action = oracle_actions.pop(0)
            if len(valid_actions) > 1:
                parser_state_l = []
                stack_embedding = stack[-1][0].view(1, self.lstm_dim)
                buffer_embedding = buffer_sent[-1] if buffer_sent else self.empty_buffer_emb    # 1 x 2*lstm_dim
                parser_state_l.append(stack_embedding)
                parser_state_l.append(buffer_embedding)
                if self.action_dim > 0:
                    parser_state_l.append(act_summary.view(1, self.lstm_dim))
                parser_state = torch.cat(parser_state_l, 1)
                h = F.relu(self.mlp(parser_state))  # 1 x lstm_dim
                if self.dropout > 0:
                    h = self.dp(h)
                f = self.act_output(h)  # 1 x out_dim
                log_probs = F.log_softmax(f, dim=1)
                cur_loss_act = - log_probs[0, action.data[0]]

                if loss_act is None:
                    loss_act = cur_loss_act
                else:
                    loss_act += cur_loss_act

            if self.action_dim > 0:
                act_embedding = F.relu(self.act_proj(self.act_embedding(action).view(1, self.action_dim))).view(1, 1, self.action_dim)  # 1 x 1 x dim
                act_summary, act_state = self.act_rnn(act_embedding, act_state)

            # execute the action to update the parser state
            if action.data[0] == self.SHIFT:
                buffer_sent.pop()
                token_embedding = input_sent.pop()  # 1 x lstm_dim
                stack_state = stack[-1][1] if len(stack) > 0 else self.stack_initial_state
                output, stack_state = self.stack_rnn(token_embedding.view(1, 1, self.lstm_dim), stack_state)
                stack.append((output, stack_state))
            else:
                right = stack.pop()
                left = stack.pop()
                head, modifier = (left, right) if action.data[0] == self.REDUCE_R else (right, left)
                top_stack_state = stack[-1][1] if len(stack) > 0 else self.stack_initial_state
                head_rep = head[0]
                mod_rep = modifier[0]
                composed_rep = F.relu(self.comp(torch.cat([head_rep, mod_rep], 2) \
                                    .view(1, 2 * self.lstm_dim)))
                output, top_stack_state = self.stack_rnn(composed_rep.view(1, 1, self.lstm_dim), top_stack_state)
                stack.append((output, top_stack_state))

        return loss_act

    def forward(self, words, pos_tags, sample=False):
        """
            parse
            Arguments:
                words(Variable):
                pos_tags(Variable):
                sample(bool): parse if Ture; sample if False
            Return:
                loss_act(Variable):
                act_sequence([]):
        """
        act_sequence = []

        if self.gpu_id > -1:
            word_inv_idx = Variable(torch.arange(words.size(1) - 1, -1, -1).long().cuda())
            pos_inv_idx = Variable(torch.arange(pos_tags.size(1) - 1, -1, -1).long().cuda())
        else:
            word_inv_idx = Variable(torch.arange(words.size(1) - 1, -1, -1).long())
            pos_inv_idx = Variable(torch.arange(pos_tags.size(1) - 1, -1, -1).long())
        # get token_embeddings and buffer (in reversed order)
        input_sent, buffer_sent=self.encode_sentence(words.index_select(1, word_inv_idx),
                                                      pos_tags.index_select(1, pos_inv_idx))
        stack = []  # stack LSTM
        if self.action_dim > 0:
            act_summary = self.empty_act_emb    # output of act rnn
            act_state = self.act_initial_state  # state of act rnn

        loss_act = None

        while not (len(stack) == 1 and len(buffer_sent) == 0):
            valid_actions = []  # based on parser state, get valid actions

            if len(buffer_sent) > 0:
                valid_actions += [self.SHIFT]
            if len(stack) >= 2:
                valid_actions += [self.REDUCE_L, self.REDUCE_R]

            action = valid_actions[0]

            if len(valid_actions) > 1:
                parser_state_l = []
                stack_embedding = stack[-1][0].view(1, self.lstm_dim)
                # 1 x 2*lstm_dim
                buffer_embedding = buffer_sent[-1] if buffer_sent else self.empty_buffer_emb
                parser_state_l.append(stack_embedding)
                parser_state_l.append(buffer_embedding)
                if self.action_dim > 0:
                    parser_state_l.append(act_summary.view(1, self.lstm_dim))
                parser_state = torch.cat(parser_state_l, 1)
                h = F.relu(self.mlp(parser_state))  # 1 x lstm_dim
                if self.dropout > 0:
                    h = self.dp(h)
                f = self.act_output(h)  # 1 x out_dim
                log_probs = F.log_softmax(f, dim=1)

                probs = torch.exp(log_probs * 0.8).data  # 1 x len_act
                for act in (self.SHIFT, self.REDUCE_L, self.REDUCE_R):
                    if act not in valid_actions:
                        probs[0, act] = 0.
                if sample:
                    action = torch.multinomial(probs, 1, replacement=True)[0,0]
                else:
                    action = torch.max(probs, 1)[1][0]  # int
                assert action in valid_actions

                cur_loss_act = - log_probs[0, action]
                if loss_act is None:
                    loss_act = cur_loss_act
                else:
                    loss_act += cur_loss_act

            if self.action_dim > 0:
                if self.gpu_id > -1:
                    act_idx = Variable(torch.LongTensor([[action]]).cuda())
                else:
                    act_idx = Variable(torch.LongTensor([[action]]))
                act_embedding = F.relu(self.act_proj(self.act_embedding(act_idx).view(1, self.action_dim)) \
                    .view(1, 1, self.action_dim))   # 1 x 1 x dim
                act_summary, act_state = self.act_rnn(act_embedding, act_state)

            # execute the action to update the parser state
            if action == self.SHIFT:
                buffer_sent.pop()
                token_embedding = input_sent.pop()  # 1 x lstm_dim
                stack_state = stack[-1][1] if len(stack) > 0 else self.stack_initial_state
                output, stack_state = self.stack_rnn(token_embedding.view(1, 1, self.lstm_dim), stack_state)
                stack.append((output, stack_state))
            else:
                right = stack.pop()
                left = stack.pop()
                head, modifier = (left, right) if action == self.REDUCE_R else (right, left)
                top_stack_state = stack[-1][1] if len(stack) > 0 else self.stack_initial_state
                head_rep = head[0]
                mod_rep = modifier[0]
                composed_rep = F.relu(self.comp(torch.cat([head_rep, mod_rep], 2).view(1, 2 * self.lstm_dim)))
                output, top_stack_state = self.stack_rnn(composed_rep.view(1, 1, self.lstm_dim), top_stack_state)
                stack.append((output, top_stack_state))
            act_sequence.append(action)

        return loss_act, act_sequence

    def parse_pr(self, words, pos_tags, rule2i, sample=False):
        """
        parse for posterior regularization
        Arguments:
            words(Variable):
            pos_tags(Variable):
            rule2i(dict):
            sample(bool): parse if Ture; sample if False
        Return:
            loss_act(Variable):
            act_sequence([]):
            feature(FloatTensor):
        """
        act_sequence = []

        n = len(rule2i)
        feature = torch.zeros(n)

        tokens = [(words.data[0,i], pos_tags.data[0,i])  for i in range(words.size(1))]
        tokens.reverse()

        stack = []      # stack LSTM
        if self.action_dim > 0:
            act_summary = self.empty_act_emb    # output of act rnn
            act_state = self.act_initial_state  # state of act rnn

        if self.gpu_id > -1:
            word_inv_idx = Variable(torch.arange(words.size(1) - 1, -1, -1).long().cuda())
            pos_inv_idx = Variable(torch.arange(pos_tags.size(1) - 1, -1, -1).long().cuda())
        else:
            word_inv_idx = Variable(torch.arange(words.size(1) - 1, -1, -1).long())
            pos_inv_idx = Variable(torch.arange(pos_tags.size(1) - 1, -1, -1).long())
        # get token_embeddings and buffer (in reversed order)
        input_sent, buffer_sent = self.encode_sentence(words.index_select(1, word_inv_idx),
                                                      pos_tags.index_select(1, pos_inv_idx))
        loss_act = None

        while not (len(stack) == 1 and len(buffer_sent) == 0):
            valid_actions = []      # based on parser state, get valid actions

            if len(buffer_sent) > 0:
                valid_actions += [self.SHIFT]
            if len(stack) >= 2:
                valid_actions += [self.REDUCE_L, self.REDUCE_R]
            
            action = valid_actions[0]

            if len(valid_actions) > 1:
                parser_state_l = []
                stack_embedding = stack[-1][0].view(1, self.lstm_dim)
                buffer_embedding = buffer_sent[-1] if buffer_sent else self.empty_buffer_emb    # 1 x 2*lstm_dim
                parser_state_l.append(stack_embedding)
                parser_state_l.append(buffer_embedding)
                if self.action_dim > 0:
                    parser_state_l.append(act_summary.view(1, self.lstm_dim))
                parser_state = torch.cat(parser_state_l, 1)
                h = F.relu(self.mlp(parser_state))      # 1 x lstm_dim
                if self.dropout > 0:
                    h = self.dp(h)
                f = self.act_output(h)  # 1 x out_dim
                log_probs = F.log_softmax(f, dim=1)

                probs = torch.exp(log_probs * 0.8).data  # 1 x len_act
                for act in (self.SHIFT, self.REDUCE_L, self.REDUCE_R):
                    if act not in valid_actions:
                        probs[0, act] = 0.
                if sample:
                    action = torch.multinomial(probs, 1, replacement=True)[0][0]
                else:
                    action = torch.max(probs, 1)[1][0]  # int
                assert action in valid_actions

                cur_loss_act = - log_probs[0, action]
                if loss_act is None:
                    loss_act = cur_loss_act
                else:
                    loss_act += cur_loss_act

            if self.action_dim > 0:
                if self.gpu_id > -1:
                    act_idx = Variable(torch.LongTensor([[action]]).cuda())
                else:
                    act_idx = Variable(torch.LongTensor([[action]]))
                act_embedding = F.relu(self.act_proj(self.act_embedding(act_idx).view(1, self.action_dim)). \
                    view(1, 1, self.action_dim))    # 1 x 1 x dim
                act_summary, act_state = self.act_rnn(act_embedding, act_state)

            # execute the action to update the parser state
            if action == self.SHIFT:
                token = tokens.pop()
                buffer_sent.pop()
                token_embedding = input_sent.pop()  # 1 x lstm_dim
                stack_state = stack[-1][1] if len(stack) > 0 else self.stack_initial_state
                output, stack_state = self.stack_rnn(token_embedding.view(1, 1, self.lstm_dim), stack_state)
                stack.append((output, stack_state, token))
            else:
                right = stack.pop()
                left = stack.pop()
                head, modifier = (
                    left, right) if action == self.REDUCE_R else (right, left)
                top_stack_state = stack[-1][1] if stack else self.stack_initial_state
                head_rep, head_tok = head[0], head[2]
                mod_rep, mod_tok = modifier[0], modifier[2]
                composed_rep = F.relu(self.comp(torch.cat([head_rep, mod_rep], 2).view(1, 2 * self.lstm_dim)))
                output, top_stack_state = self.stack_rnn(composed_rep.view(1, 1, self.lstm_dim), top_stack_state)
                stack.append((output, top_stack_state, head_tok))

                head_pos = self.i2w_pos[head_tok[1]]
                mod_pos = self.i2w_pos[mod_tok[1]]
                if (head_pos, mod_pos) in rule2i:
                    # print (head_pos, mod_pos)
                    feature[rule2i[(head_pos, mod_pos)]] -= 1

            act_sequence.append(action)

        tok_mod = stack.pop()
        mod_pos = self.i2w_pos[tok_mod[2][1]]
        head_pos = 'ROOT'
        if (head_pos, mod_pos) in rule2i:
            # print (head_pos, mod_pos)
            feature[rule2i[(head_pos, mod_pos)]] -= 1

        return loss_act, act_sequence, feature
