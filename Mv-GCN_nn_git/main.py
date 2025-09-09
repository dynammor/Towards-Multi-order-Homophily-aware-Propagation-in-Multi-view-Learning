from __future__ import division
from __future__ import print_function

import os
from warnings import simplefilter
import numpy as np
import torch
from scipy.io import savemat

from MvGCN import EnhancedNetwork, MultiViewRevise
from MvMLP import MvMLP
from trian_mlp import train, test, Ftmodel
# from train2 import train, test, Ftmodel
from DataLoader import LoadMatData
import random
import datetime
from args import parameter_parser

import os


# from utils import plot_3d_surface_vector

os.environ['OMP_NUM_THREADS'] = '1'

simplefilter(action='ignore', category=FutureWarning)

args = parameter_parser()
args.cuda = not args.no_cuda and torch.cuda.is_available()

device = torch.device('cpu' if args.device == 'cpu' else 'cuda:' + args.device)

# for dataset in [  ]:
for dataset in ['100leaves']:
# for dataset in ['scene15', 'Out_Scene', 'NoisyMNIST_15000']:
# for dataset in ['animals']:

    print("============================================{}================================================".format(
        dataset))
    # 'Caltech101-20', 'Caltech101-all', 'citeseer', 'COIL', 'flower17', 'GRAZ02', 'handwritten', 'Hdigit', 'HW', 'Mfeat', 'MNIST10k', 'MITIndoor',
# 'MSRC-v1', 'NoisyMNIST_15000', 'NUS-WIDE', '3sources', '20newsgroups', '100leaves', 'ALOI', 'animals', 'BBCnews', 'BBCSports', 'Caltech101-7',
    #  'COIL',  'GRAZ02','Notting-Hill', 'ORL2', 'scene15', 'UCI', 'WebKB',
    #  'Wikipedia', 'Youtube','AwA','NUSWIDEOBJ'
    #  'Reuters'报错
    args.dataset = dataset
    # /public/home/wsp/inspur/wjy/dataset//
    adj_ori, features_ori, labels_ori, nfeats_ori, num_view_ori, num_class_ori,adj_noself_ori = LoadMatData(args.dataset, args.k,
                                                                     'D:\project\dataset//')
    neighbors = []

    for i, a in enumerate(adj_ori):
        adj_indices = a._indices()

        num_nodes = a.shape[0]

        matrix_neighbors = [[] for _ in range(num_nodes)]

        for j in range(adj_indices.shape[1]):
            src = adj_indices[0, j].item()
            dst = adj_indices[1, j].item()
            matrix_neighbors[src].append(dst)

            matrix_neighbors[dst].append(src)

        neighbors.append(matrix_neighbors)

    # for epoch in [300]:
    for epoch in [1000]:
        pa_alpha=[]
        pa_U=[]
        pa_ACC = np.zeros((9, 9))
        ration_acc=[]
        # for a in [1,2,3,4,5,6,7,8,9]:
        for a in [5]:
            pa_alpha.append(a)
            # for epoch_2 in [500]:
            for epoch_2 in [1000]:
                for k in [5]:
                # for k in [1,2,3,4,5,6,7,8,9]:
                # for r in range(5, 51, 5):
                #     k=5
                #     args.train_ratio=r*0.01

                    pa_U.append(k)
                    for lr in [0.001]:
                        for hid in [32]:
                                np.random.seed(args.seed)
                                torch.manual_seed(args.seed)
                                if args.cuda:
                                    torch.cuda.manual_seed(args.seed)
                                    torch.cuda.manual_seed_all(args.seed)
                                random.seed(args.seed)
                                torch.backends.cudnn.deterministic = True
                                torch.backends.cudnn.benchmark = False
                                torch.backends.cudnn.enabled = True

                                args.lr = lr
                                args.nhid = hid
                                args.epoch = epoch
                                args.rep_num = 1
                                if dataset=='animals':
                                    a=7
                                    k=4
                                elif dataset=='100leaves':
                                    a=5
                                    k=6
                                elif dataset=='MNIST10k':
                                    a=5
                                    k=6

                                elif dataset=='MSRC-v1':
                                    a=4
                                    k=3

                                elif dataset=='flower17':
                                    a=6
                                    k=3

                                elif dataset=='GRAZ02':
                                    a=8
                                    k=3
                                elif dataset=='NoisyMNIST_15000':
                                    a=5
                                    k=6

                                elif dataset=='NUS-WIDE':
                                    a=3
                                    k=5

                                elif dataset=='Out_Scene':
                                    a=4
                                    k=5

                                elif dataset=='scene15':
                                    a=6
                                    k=3

                                acc_pre = []
                                f1_pre = []
                                acc = []
                                f1 = []
                                acc_300s= []
                                acc_500s= []
                                NoWF_acc = []
                                NoWF_f1 = []
                                Nocer_acc = []
                                Nocer_f1 = []
                                Noun_acc = []
                                Noun_f1 = []
                                for i in range(args.rep_num):
                                    args.layer_num = 2
                                    with torch.cuda.device(device):
                                        torch.cuda.empty_cache()

                                    adj=adj_ori.copy()
                                    features=features_ori.copy()
                                    labels=labels_ori
                                    nfeats=nfeats_ori.copy()
                                    num_view=num_view_ori
                                    num_class=num_class_ori
                                    adj_noself=adj_noself_ori
                                    print("===============================rep_num:{}=================================".format(i))
                                    # adj, features, labels, nfeats, num_view, num_class = LoadMatData(args.dataset, args.k,
                                    #                                                                  'D:\project\dataset//')
                                    print("dataset loading finished, num_view: {}, num_feat: {}, num_class: {}".format(num_view,
                                                                                                                       nfeats,
                                                                                                                       num_class))
                                    print('rep_num:', i + 1)
                                    model, features, labels, adj, idx_test, output, idx_train = train(adj, features, labels, nfeats,
                                                                                                      num_view, num_class, args,
                                                                                                      device,0,args.layer_num)

                                    model_noself_twoloop,_, _, adj_noself,_, output,_ = train(adj_noself, features,
                                                                                                      labels, nfeats,
                                                                                                      num_view,
                                                                                                      num_class, args,
                                                                                                      device,1,args.layer_num)
                                    args.layer_num=args.layer_num-1
                                    model_noself_oneloop, _, _, adj_noself, _, output, _ = train(adj_noself, features,
                                                                                                 labels, nfeats,
                                                                                                 num_view,
                                                                                                 num_class, args,
                                                                                                 device, 1,args.layer_num-1)

                                    view_dims = [x.shape[1] for x in features]

                                    # Testing
                                    Enhancednetwork = EnhancedNetwork(num_class).to(device)
                                    MultiViewrevise = MvMLP(nfeats, num_class).to(device)
                                    # MultiViewrevise = MultiViewRevise(view_dims, args.nhid, num_class).to(device)

                                    acc_test_pre, f1_score_pre = test(args, model, features, labels, adj, idx_test)
                                    acc_test, f1_score, flag, NoWF_acc_test, NoWF_f1_score, Nocer_acc_test, Nocer_f1_score, Noun_acc_test, Noun_f1_score, correct_prediction_probability,acc_300,acc_500 = Ftmodel(
                                        args, model, features, labels, adj, idx_test, Enhancednetwork, MultiViewrevise, num_class,
                                        device, idx_train, k, epoch_2, nfeats,a,neighbors,model_noself_twoloop,model_noself_oneloop)

                                    acc_pre.append(acc_test_pre)
                                    f1_pre.append(f1_score_pre)
                                    acc.append(acc_test)
                                    f1.append(f1_score)
                                    acc_300s.append(acc_300)
                                    acc_500s.append(acc_500)
                                    NoWF_acc.append(NoWF_acc_test)
                                    NoWF_f1.append(NoWF_f1_score)
                                    Nocer_acc.append(Nocer_acc_test)
                                    Nocer_f1.append(Nocer_f1_score)
                                    Noun_acc.append(Noun_acc_test)
                                    Noun_f1.append(Noun_f1_score)


                                    del Enhancednetwork
                                    del MultiViewrevise

                                    # if flag == 1:
                                    #     break

                                print("Optimization Finished!")

                                print("pre_accuracy_mean= {:.4f}".format(np.array(acc_pre).mean()),
                                      "accuracy_std= {:.4f}".format(np.array(acc_pre).std()))
                                print("pre_f1_mean= {:.4f}".format(np.array(f1_pre).mean()),
                                      "f1_std= {:.4f}".format(np.array(f1_pre).std()))

                                print("accuracy_mean= {:.4f}".format(np.array(acc).mean()),
                                      "accuracy_std= {:.4f}".format(np.array(acc).std()))
                                print("f1_mean= {:.4f}".format(np.array(f1).mean()), "f1_std= {:.4f}".format(np.array(f1).std()))

                                isExists = os.path.exists(args.res_path)
                                if not isExists:
                                    os.mkdir(args.res_path)


                                with open(args.res_path + '/{}.txt'.format(args.dataset), 'a', encoding='utf-8') as f:
                                    f.write(datetime.datetime.now().strftime('%Y-%m-%d  %H:%M:%S') + '\n'
                                                                                                     'dataset:{} | layer_num:{} | rep_num：{} | Ratio：{}'.format(
                                        args.dataset, args.layer_num, args.rep_num, args.train_ratio) + '\n'
                                                                                                        'dropout:{} | epochs:{} | lr:{} | wd:{} | hidden:{}'.format(
                                        args.dropout, args.epoch, args.lr, args.weight_decay, args.nhid) + '\n'
            
                                                                                                           'ACC_pre_mean:{:.4f} | ACC_pre_std: {:.4f} | ACC_pre_max:{:.4f}'.format(
                                        np.array(acc_pre).mean(), np.array(acc_pre).std(), np.array(acc_pre).max()) + '\n'
                                                                                                                      'F1_pre_mean:{:.4f} | F1_pre_std: {:.4f} | F1_pre_max:{:.4f}|k:{:.4f}'.format(
                                        np.array(f1_pre).mean(), np.array(f1_pre).std(), np.array(f1_pre).max(), k) + '\n'
            
                                                                                                                      'ACC_mean_1000:{:.4f} | ACC_std: {:.4f} | ACC_max:{:.4f}'.format(
                                        np.array(acc).mean(), np.array(acc).std(), np.array(acc).max()) + '\n'
                                                                                                          'ACC_mean_300:{:.4f} | ACC_mean_500:{:.4f}'.format(
                                        np.array(acc_300s).mean(),  np.array(acc_500s).mean()) + '\n'
                                                                                                          'F1_mean:{:.4f} | F1_std: {:.4f} | F1_max:{:.4f}|k:{:.4f}'.format(
                                        np.array(f1).mean(), np.array(f1).std(), np.array(f1).max(), k) + '\n'
            
                                                                                                          'NoWF_ACC_mean:{:.4f} | NoWF_ACC_std: {:.4f} | NoWF_ACC_max:{:.4f}'.format(
                                        np.array(NoWF_acc).mean(), np.array(NoWF_acc).std(), np.array(NoWF_acc).max()) + '\n'
                                                                                                                         'NoWF_F1_mean:{:.4f} | NoWF_F1_std: {:.4f} | NoWF_F1_max:{:.4f}|k:{:.4f}'.format(
                                        np.array(NoWF_f1).mean(), np.array(NoWF_f1).std(), np.array(NoWF_f1).max(), k) + '\n'
            
                                                                                                                         'Nocer_ACC_mean:{:.4f} | Nocer_ACC_std: {:.4f} | Nocer_ACC_max:{:.4f}'.format(
                                        np.array(Nocer_acc).mean(), np.array(Nocer_acc).std(), np.array(Nocer_acc).max()) + '\n'
                                                                                                                            'Nocer_F1_mean:{:.4f} | Nocer_F1_std: {:.4f} | Nocer_F1_max:{:.4f}|k:{:.4f}'.format(
                                        np.array(Nocer_f1).mean(), np.array(Nocer_f1).std(), np.array(Nocer_f1).max(), k) + '\n'
            
                                                                                                                            'Noun_ACC_mean:{:.4f} | Noun_ACC_std: {:.4f} | Noun_ACC_max:{:.4f}'.format(
                                        np.array(Noun_acc).mean(), np.array(Noun_acc).std(), np.array(Noun_acc).max()) + '\n'
                                                                                                                         'Noun_F1_mean:{:.4f} | Noun_F1_std: {:.4f} | Noun_F1_max:{:.4f}|k:{:.4f}|a:{:.4f}|correct_prediction_probability:{:.4f}|E2:{:.4f}'.format(
                                        np.array(Noun_f1).mean(), np.array(Noun_f1).std(), np.array(Noun_f1).max(), k,a,
                                        correct_prediction_probability,epoch_2) + '\n'
                                                                          '----------------------------------------------------------------------------' + '\n')

                                # pa_ACC[a-1][k-1]=np.array(acc).mean()
                                ration_acc.append(np.array(acc).mean())

                                del model
                                del model_noself_oneloop
                                del model_noself_twoloop


        data_dict = {
            'acc_ration': ration_acc
        }
        savemat(f'./results/mat/{args.dataset}_ration.mat', data_dict)

        # data_dict = {
        #     'acc_matrix': pa_ACC
        # }
        # savemat(f'./results/mat/{args.dataset}_epoch_{epoch}_{epoch_2}.mat', data_dict)


    print("============================================{}================================================".format(
        dataset))