import math
import geoopt as gt
from geoopt.manifolds import PoincareBall
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


class GraphConvolution(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
            + str(self.in_features) + ' -> ' \
            + str(self.out_features) + ')'

class GCN_onehop(nn.Module):
    """GCN Network Structure"""
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GCN, self).__init__()

        self.layer_list = nn.ModuleList()

        self.gc_in = GraphConvolution(nfeat, nclass)
        self.gc_in2 = nn.Linear(nhid, nhid)
        self.gc_out =  nn.Linear(nhid, nclass)
        self.dropout = dropout
        self.reset_parameters()

    def reset_parameters(self):
        self.gc_in.reset_parameters()
        self.gc_out.reset_parameters()

    def forward(self, x, adj):
        x = self.gc_in(x, adj)
        x=F.relu(self.gc_in2(x))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc_out(x)
        return F.log_softmax(x, dim=1)

class GCN(nn.Module):
    """GCN Network Structure"""

    def __init__(self, nfeat, nhid, nclass, dropout,layers=args.layer_num):
        super(GCN, self).__init__()

        self.layer_list = nn.ModuleList()

        self.gc_in = GraphConvolution(nfeat, nhid)
        for _ in range(layers - 2):
            self.layer_list.append(GraphConvolution(nhid, nhid))
        # self.gc_out = GraphConvolution(nhid, nclass)
        self.gc_out = GraphConvolution(nhid, nclass)
        self.dropout = dropout

        self.reset_parameters()

    def reset_parameters(self):
        self.gc_in.reset_parameters()
        for layer in self.layer_list:
            layer.reset_parameters()
        self.gc_out.reset_parameters()

    def forward(self, x, adj):
        x = F.relu(self.gc_in(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)

        for layer in self.layer_list:
            x = layer(x, adj)

        x = self.gc_out(x, adj)
        return F.log_softmax(x, dim=1)


class MvGCN(nn.Module):
    def __init__(self, nfeats, num_class, nhid=args.nhid, dropout=args.dropout, batch_norm=1):
        super(MvGCN, self).__init__()
        self.dropout = dropout
        self.batch_norm = batch_norm
        self.GCNs = torch.nn.ModuleList()
        self.fc = nn.Linear(nhid,num_class )
        for nfeat in nfeats:
            self.GCNs.append(GCN(nfeat, nhid, num_class, args.dropout))

        if self.batch_norm:
            self.bn1 = nn.BatchNorm1d(num_class)
        num_of_view = len(nfeats)
        self.W = nn.Parameter(torch.randn(num_of_view, 1), requires_grad=True)
        self.reset_parameters()
        self.manifold = gt.PoincareBall(c=1)

    def reset_parameters(self):
        for i in range(len(self.GCNs)):
            self.GCNs[i].reset_parameters()
        nn.init.kaiming_uniform_(self.W, a=math.sqrt(5))

    def forward(self, X, adj):
        GCN_outputs = []
        for idx, model in enumerate(self.GCNs):


            tmp_output = model(X[idx], adj[idx])

            # x = tmp_output.float()
            # # x_hyp = self.manifold.expmap0(x)
            # x_hyp = self.manifold.expmap0(self.manifold.logmap0(self.fc(x)))


            GCN_outputs.append(tmp_output)

        output = torch.stack(GCN_outputs, dim=1)  # (n, len(ks), nfeat)


        # output = F.normalize(output, dim=-1)
        # W = F.softmax(self.W)
        output = F.softmax(self.W,dim=1) * output

        output = output.sum(1)
        ori_output = output
        if self.batch_norm:
            output = self.bn1(output)
        output = F.dropout(output, self.dropout, training=self.training)

        return output, ori_output, F.softmax(ori_output,dim=1),F.log_softmax(ori_output, dim=1),GCN_outputs

class MvGCN_onehop(nn.Module):
    def __init__(self, nfeats, num_class, nhid=args.nhid, dropout=args.dropout, batch_norm=1):
        super(MvGCN, self).__init__()
        self.dropout = dropout
        self.batch_norm = batch_norm
        self.GCNs = torch.nn.ModuleList()

        for nfeat in nfeats:
            self.GCNs.append(GCN_onehop(nfeat, nhid, num_class, args.dropout))

        if self.batch_norm:
            self.bn1 = nn.BatchNorm1d(num_class)
        num_of_view = len(nfeats)
        self.W = nn.Parameter(torch.randn(num_of_view, 1), requires_grad=True)
        self.reset_parameters()

    def reset_parameters(self):
        for i in range(len(self.GCNs)):
            self.GCNs[i].reset_parameters()
        nn.init.kaiming_uniform_(self.W, a=math.sqrt(5))

    def forward(self, X, adj):
        GCN_outputs = []
        for idx, model in enumerate(self.GCNs):
            tmp_output = model(X[idx], adj[idx])
            GCN_outputs.append(tmp_output)

        output = torch.stack(GCN_outputs, dim=1)  # (n, len(ks), nfeat)

        output = F.normalize(output, dim=-1)
        # W = F.softmax(self.W)
        output = F.softmax(self.W,dim=1) * output

        output = output.sum(1)
        ori_output = output
        if self.batch_norm:
            output = self.bn1(output)
        output = F.dropout(output, self.dropout, training=self.training)

        return output, ori_output, F.softmax(ori_output,dim=1),F.log_softmax(ori_output, dim=1),GCN_outputs

class EnhancedNetwork(nn.Module):
    def __init__(self, latent_dim):
        super(EnhancedNetwork, self).__init__()
        self.fc1 = nn.Linear(latent_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, latent_dim)

    def forward(self, z):
        z = F.relu(self.fc1(z))
        z = F.relu(self.fc2(z))
        score = self.fc3(z)
        return F.log_softmax(score, dim=1)


class MultiViewRevise(nn.Module):
    def __init__(self, view_dims, hidden_dim, output_dim):
        super(MultiViewRevise, self).__init__()
        self.num_views = len(view_dims)

        self.view_mappers = nn.ModuleList([nn.Linear(dim, hidden_dim) for dim in view_dims])

        self.Revise_layers = nn.Sequential(
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, views):
        view_features = []
        for i in range(self.num_views):
            mapped_feature = self.view_mappers[i](views[i])
            feature = self.Revise_layers(mapped_feature)
            view_features.append(feature)

        fused_features = torch.stack(view_features, dim=0).mean(dim=0)

        return F.log_softmax(fused_features, dim=1)
