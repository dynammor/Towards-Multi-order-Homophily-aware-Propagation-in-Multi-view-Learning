from __future__ import division
from __future__ import print_function

from warnings import simplefilter
import numpy as np
import torch

from DataLoader import LoadMatData
import random
from args import parameter_parser
import os

os.environ['OMP_NUM_THREADS'] = '1'

simplefilter(action='ignore', category=FutureWarning)

args = parameter_parser()
args.cuda = not args.no_cuda and torch.cuda.is_available()

device = torch.device('cpu' if args.device == 'cpu' else 'cuda:' + args.device)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
random.seed(args.seed)
# 禁用 cuDNN 的随机性（重要）
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.enabled = True


def calculate_homophily_rate(adj_matrix, labels):


    if adj_matrix.is_sparse:
        adj_matrix = adj_matrix.to_dense()


    row_indices, col_indices = torch.nonzero(adj_matrix, as_tuple=True)


    total_edges = 0
    same_label_edges = 0


    for i, j in zip(row_indices, col_indices):
        if i != j:
            total_edges += 1
            if labels[i] == labels[j]:
                same_label_edges += 1


    homophily_rate = same_label_edges / total_edges if total_edges > 0 else 0.0
    return homophily_rate

# for dataset in ['COIL','Caltech101-20', 'Caltech101-all', 'citeseer', 'COIL', 'GRAZ02', 'handwritten', 'Hdigit', 'HW', 'Mfeat', 'MNIST10k',
#                 'MITIndoor', 'MSRC-v1', 'NoisyMNIST_15000', 'NUS-WIDE', '3sources', '20newsgroups', '100leaves', 'ALOI', 'animals',
#                 'BBCnews', 'BBCSports', 'Caltech101-7', 'NUS-WIDE','scene15',  'GRAZ02','Notting-Hill', 'ORL2']:
# for dataset in ['flower17','Wikipedia', 'Youtube','NUSWIDEOBJ', 'WebKB', 'UCI']:
for dataset in ["MNIST10k","NoisyMNIST_15000","MSRC-v1","GRAZ02","NUS-WIDE","100leaves","animals","scene15","flower17","Out_Scene"]:

    print("============================================{}================================================".format(
        dataset))

    args.dataset = dataset
    for k in [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40]:
        args.k=k
        adj_ori, features_ori, labels_ori, nfeats_ori, num_view_ori, num_class_ori,adj_noself_ori = LoadMatData(args.dataset, args.k,
                                                                         'D:\project\dataset//')
        adj = adj_ori.copy()
        features = features_ori.copy()
        labels = labels_ori
        nfeats = nfeats_ori.copy()
        num_view = num_view_ori
        num_class = num_class_ori
        adj_noself = adj_noself_ori
        sum=0
        for i in range(len(adj)):
            h=calculate_homophily_rate(adj[i],labels)
            sum=sum+h
        print(args.k,':',sum/len(adj))

