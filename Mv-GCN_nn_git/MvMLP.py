import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
import torch.sparse as sparse
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
from args import parameter_parser

args = parameter_parser()


class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, dropout=0.5, batch_norm=False):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.tanh = nn.Tanh()
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.dropout = dropout
        self.batch_norm = batch_norm
        if self.batch_norm:
            self.bn1 = nn.BatchNorm1d(hidden_size)

    def forward(self, x):
        out = self.fc1(x)
        if self.batch_norm:
            out = self.bn1(out)
        out = self.tanh(out)
        out = F.dropout(out, self.dropout, training=self.training)
        out = self.fc2(out)
        return out

    def reset_parameters(self):
        self.fc1.reset_parameters()
        self.fc2.reset_parameters()


class MvMLP(nn.Module):
    def __init__(self, nfeats, nclass, nhid=args.nhid, dropout=0.5, batch_norm=False):
        super(MvMLP, self).__init__()
        self.dropout = dropout
        self.batch_norm = batch_norm
        self.MLPs = torch.nn.ModuleList()
        for nfeat in nfeats:
            self.MLPs.append(MLP(nfeat, nhid, nclass))
        num_of_view = len(nfeats)
        self.W = nn.Parameter(torch.randn(num_of_view, 1), requires_grad=True)
        self.reset_parameters()
        if self.batch_norm:
            self.bn1 = nn.BatchNorm1d(nclass)

    def reset_parameters(self):
        for i in range(len(self.MLPs)):
            self.MLPs[i].reset_parameters()
        nn.init.kaiming_uniform_(self.W, a=math.sqrt(5))
    def forward(self, X):
        MLP_outputs = []
        for idx, model in enumerate(self.MLPs):
            tmp_output = model(X[idx])  # 每个视角MLP
            MLP_outputs.append(tmp_output)
        output = torch.stack(MLP_outputs, dim=1)  # (n, len(ks), nfeat)
        # output = F.normalize(output, dim=-1)
        output = self.W * output
        output = output.sum(1)
        if self.batch_norm:  # 加了不好
            output = self.bn1(output)
        output = F.dropout(output, self.dropout, training=self.training)
        return output
