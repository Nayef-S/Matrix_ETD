import sys
import math
import numpy as np
import torch
import scipy.linalg
from torch import nn
from torch.nn import init
from torch.autograd import Variable
from logm import logm
from torch_ETD import ETD1_BCH1, ETD1_BCH2 ,ETD2
import torch.nn.functional as F


def etd_integration(IC, m1_L, m1_R, m2, N, dt, t):
    n_steps = int(t[-1] // dt)

    for _ in range(n_steps):
        linear_part = m1_L @ IC @ m1_R
        non_linear_part = m2 @ N
        IC = linear_part + non_linear_part

    return IC



class ODEblockW(nn.Module):
    def __init__(self, in_features, out_features, opt, adj, deg, t=torch.tensor([0,1])):
        super(ODEblockW, self).__init__()
        self.t = t
        self.opt = opt
        self.adj = adj
        self.dt_ = opt['dt']
        #self.dt_train =  nn.Parameter(torch.tensor(self.dt_))
        self.x0 = None
        self.nfe = 0
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = opt['alpha']
        self.alpha_train = nn.Parameter(self.alpha*torch.ones(adj.shape[1]))
        self.nfe = 0

        self.w = nn.Parameter(torch.eye(2*opt['hidden_dim']))
        self.d = nn.Parameter(torch.zeros(2*opt['hidden_dim']) + 1)

    def set_x0(self, x0):
        self.x0 = x0.clone().detach()

    def forward(self, x):
        self.nfe +=1

        alph = F.sigmoid(self.alpha_train).unsqueeze(dim=1)
        #dt = torch.clamp(self.dt_train, min=0)
        
        d = torch.clamp(self.d, min=0, max=1)
        w = torch.mm(self.w * d, torch.t(self.w))

        diff = self.x0.shape[0] - w.shape[0]

        m = nn.ConstantPad2d((0,diff,0,diff), 0)

        w_padded = m(w)

        commute_A_W = torch.mm(self.adj.to_dense(),w_padded) - torch.mm(w_padded,self.adj.to_dense())

        t = self.t.type_as(x)

        m1_L , m1_R , m2  = ETD1_BCH2(alph * 0.5 * (self.adj.to_dense()) -  alph * 0.5 * torch.eye(self.adj.shape[0]).cuda(), (w - torch.eye(w.shape[0]).cuda()), self.dt_)

        z = etd_integration(x, m1_L , m1_R , m2 , self.x0, self.dt_, self.t)

        return z

    def __repr__(self):
        return self.__class__.__name__ + '( Time Interval ' + str(self.t[0].item()) + ' -> ' + str(self.t[1].item()) \
               + ")"


# Define the GNN model.
class WGNN(nn.Module):
    def __init__(self, opt, adj, deg, time):
        super(WGNN, self).__init__()
        self.opt = opt
        self.adj = adj
        self.T = time

        self.m1 = nn.Linear(opt['num_feature'], opt['hidden_dim'])

        self.odeblock = ODEblockW(opt['hidden_dim'], opt['hidden_dim'], opt, adj, deg, t=torch.tensor([0,self.T]))

        self.m2 = nn.Linear(opt['hidden_dim'], opt['num_class'])

        if opt['cuda']:
            self.cuda()

    def reset(self):
        self.m1.reset_parameters()
        self.m2.reset_parameters()

    def forward(self, x):
        # Encode each node based on its feature.
        x = F.dropout(x, self.opt['input_dropout'], training=self.training)
        x = self.m1(x)

        # Solve the initial value problem of the ODE.
        c_aux = torch.zeros(x.shape).cuda()
        x = torch.cat([x,c_aux], dim=1)
        self.odeblock.set_x0(x)

        z = self.odeblock(x)
        z = torch.split(z, x.shape[1]//2, dim=1)[0]

        # Activation.
        z = F.relu(z)

        # Dropout.
        z = F.dropout(z, self.opt['dropout'], training=self.training)

        # Decode each node embedding to get node label.
        z = self.m2(z)
        return z

