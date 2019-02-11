import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math

class PR(nn.Module):

    def __init__(self, epsilon, pr_fname, mc_samples, para_init=None, gpu_id=-1):
        """
        Arguments:
            epsilon(float):
            pr_fname(str):
            mc_samples(int):
        """
        super(PR, self).__init__()
        self.epsilon = epsilon
        self.rules = [] # [(head, mod), ...]
        self.mc_samples = mc_samples
        self.gpu_id = gpu_id

        b = []  # expectations (b must be negative here)
        with open(pr_fname, 'r') as f:
            for line in f:
                head, mod, expectation = line.split()
                self.rules.append((head, mod))
                b.append( - float(expectation))
        self.rule2i = {self.rules[i]: i for i in range(len(self.rules))}
        n = len(self.rules)
        M = self.mc_samples
        self.phi = torch.zeros(M, n)    # features
        if self.gpu_id > -1:
            self.phi = self.phi.cuda()
        self.phi = Variable(self.phi)
        self.Lambda = nn.Linear(n, 1, False) # no bias
        
        if self.gpu_id > -1:
            self.b = Variable(torch.FloatTensor(b).cuda())
            self.cuda()
        else:
            self.b = Variable(torch.FloatTensor(b))

        print 'initializing PR'
        if para_init is None:
            raise ValueError('No initializer')
        else:
            para_init(self.Lambda.weight)

        self.Lambda.weight = nn.Parameter(-self.b.data.clone().unsqueeze_(0))

        self.project()

    def forward(self):
        '''
        Arguments:
            phi(Variable): negative M x n
        Return:
            objective(Variable): pr loss
            pr_factor(Variable): M x 1
        '''
        M = self.phi.size(0)
        temp = - self.Lambda(self.phi) # M x 1
        temp = temp.squeeze(1)
        log_Z = - math.log(M) + log_sum_exp(temp, dim=0)
        objective = torch.mv(self.Lambda.weight, self.b) + log_Z + torch.norm(self.Lambda.weight, p=2) * self.epsilon
        pr_factor = temp - log_Z    # M
        pr_factor = torch.exp(pr_factor)    # M
        return objective, pr_factor

    def project(self):
        """
        constrain lambda is no less than zero
        """
        self.Lambda.weight.data.copy_(F.relu(self.Lambda.weight).data)

    def reset_phi(self):
        """
        reset phi
        """
        self.phi.zero_()

    def check_total(self, pos, rule_total):
        """
        count the total number of pos tag occurences which involve in rules
        Arguments:
            pos([str]): pos tags of sentences
            rule_total(dic):
        """
        for i in range(len(self.rules)):
            head, mod = self.rules[i]
            if head in pos + ['ROOT'] and mod in pos:
                rule_total[i] += 1
                continue

    # def check_occ(self, phi, rule_occ):
    #     """
    #         desperate
    #     """
    #     rule_occ -= np.sum(phi, 1)


def log_sum_exp(value, dim=None, keepdim=False):
    """
    Numerically stable implementation of the operation value.exp().sum(dim, keepdim).log()
    """
    if dim is None:
        raise ValueError('Plese specify dim')
    m, _ = torch.max(value, dim=dim, keepdim=True)
    value0 = value - m
    if keepdim is False:
        m = m.squeeze(dim)
    return m + torch.log(torch.sum(torch.exp(value0), dim=dim, keepdim=keepdim))