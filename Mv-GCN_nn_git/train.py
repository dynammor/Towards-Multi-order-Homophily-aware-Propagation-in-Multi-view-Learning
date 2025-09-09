from __future__ import division
from __future__ import print_function

import math
import time

import numpy as np

from DataLoader import LoadMatData, generate_permutation, normalize, normalize2
import torch.nn.functional as F
import torch.optim as optim
from MvMLP import MvMLP
from MvGCN import MvGCN

from utils import accuracy, f1_test
from tqdm import tqdm
import os
import torch

def train(adj, features, labels, nfeats, num_view, num_class, args, device):

    idx_train, idx_test = generate_permutation(labels, args)
    # model = MvMLP(nfeats, num_class)
    model = MvGCN(nfeats, num_class)
    # model = MvFGCN(nfeats, num_class)
    # model = MvACMGCN(nfeats, num_class)
    total_para = sum(x.numel() for x in model.parameters())
    print("Total number of paramerters in networks is {}  ".format(total_para / 1e6))
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    for i in range(num_view):
        exec("features_{}= torch.from_numpy(features[{}]/1.0).float().to(device)".format(i, i))
        exec("features[{}]= torch.Tensor(features[{}] / 1.0).to(device)".format(i, i))
        exec("features[{}] = F.normalize(features[{}])".format(i, i))
        exec("adj[{}]=adj[{}].to_dense().float().to(device)".format(i, i))

    if args.cuda:
        model.cuda()
        labels = labels.cuda()
        idx_train = idx_train.cuda()  # [:10]
        idx_test = idx_test.cuda()
    # f_loss = open('./results/loss_and_acc_curve/loss/' + args.dataset + '.txt', 'w')
    # f_ACC = open('./results/loss_and_acc_curve/ACC/' + args.dataset + '.txt', 'w')
    # f_F1 = open('./results/loss_and_acc_curve/F1/' + args.dataset + '.txt', 'w')
    t1 = time.time()

    with tqdm(total=args.epoch) as pbar:
        pbar.set_description('Training:')

        for i in range(args.epoch):
            t = time.time()
            model.train()
            optimizer.zero_grad()
            output,_,_ ,_= model(features, adj)
            output = F.log_softmax(output, dim=1)
            # temp1 = output[idx_train]
            # temp2 = labels[idx_train]
            loss_train = F.nll_loss(output[idx_train], labels[idx_train])
            acc_train = accuracy(output[idx_train], labels[idx_train])
            f1_train = f1_test(output[idx_train], labels[idx_train])
            loss_train.backward()
            # clip_gradient(model, clip_norm=0.5)
            optimizer.step()

            if not args.fastmode:
                # Evaluate validation set performance separately,
                # deactivates dropout during validation run.
                model.eval()
                output,_,_,_ = model(features, adj)
                output = F.log_softmax(output, dim=1)

            loss_test = F.nll_loss(output[idx_test], labels[idx_test])
            acc_test = accuracy(output[idx_test], labels[idx_test])
            f1 = f1_test(output[idx_test], labels[idx_test])

            # # loss曲线
            # isExists = os.path.exists("./results_linear/loss/{}".format(args.dataset))
            # if not isExists:
            #     os.mkdir("./results_linear/loss/{}".format(args.dataset))
            # with open("./results_linear/loss/{}".format(args.dataset) + '/loss.txt', 'a', encoding='utf-8') as f:
            #     f.write(str(loss_test.detach().cpu().numpy()) + '\n')
            # with open("./results_linear/loss/{}".format(args.dataset) + '/acc.txt', 'a', encoding='utf-8') as f:
            #     f.write(str(acc_test.detach().cpu().numpy()) + '\n')
            # with open("./results_linear/loss/{}".format(args.dataset) + '/f1.txt', 'a', encoding='utf-8') as f:
            #     f.write(str(f1) + '\n')

            outstr = 'Epoch: {:04d} '.format(i + 1) + \
                     'loss_train: {:.4f} '.format(loss_train.item()) + \
                     'acc_train: {:.4f} '.format(acc_train.item()) + \
                     'loss_test: {:.4f} '.format(loss_test.item()) + \
                     'acc_test: {:.4f} '.format(acc_test.item()) + \
                     'f1_test: {:.4f} '.format(f1.item()) + \
                     'time: {:.4f}s'.format(time.time() - t)
            pbar.set_postfix_str(outstr)
            # print(outstr)
            # f_loss.write(str(loss_train.item()) + '\n')
            # f_ACC.write(str(acc_test.item()) + '\n')
            # f_F1.write(str(f1.item()) + '\n')
            pbar.update(1)
    # return model, loss_val.item(), acc_val.item(), loss_test.item(), acc_test.item()
    # draw_plt(output[idx_test], labels[idx_test], args.dataset)
    print('total_time:', time.time() - t1)
    torch.save({
        'x_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, f'./checkpoints/best_{args.dataset}.pth')
    return model, features, labels, adj, idx_test, output,idx_train

def test(args, model, features, labels, adj, idx_test):
    model.eval()
    output, ori_output, sof_output ,logsof_putput = model(features, adj)

    # # TSNE
    # prediction_test = F.log_softmax(output[idx_test], dim=1)
    # labels_test = labels[idx_test]
    # prediction_test = prediction_test.detach().cpu().numpy()
    # labels_test = labels_test.detach().cpu().numpy()
    # sio.savemat('{}_MvOGCN_linear_prediction_test.mat'.format(args.dataset), {'prediction_test': prediction_test})
    # sio.savemat('{}_MvOGCN_linear_labels_test.mat'.format(args.dataset), {'labels_test': labels_test})

    acc_test = accuracy(output[idx_test], labels[idx_test])
    f1 = f1_test(output[idx_test], labels[idx_test])
    return 100 * acc_test.item(), 100 * f1.item()


def Ftmodel(args, model, features, labels, adj, idx_test,EnhancedNetwork,ReviseNetwork,num_class,device,idx_train,k,epochs):
    checkpoint = torch.load(f'./checkpoints/best_{args.dataset}.pth')
    model.load_state_dict(checkpoint['x_state_dict'])
    model.eval()

    output,ori_output,sof_output,logsof_putput = model(features, adj)
    predicted_classes = torch.argmax(output, dim=1)
    incorrect_indices = (predicted_classes != labels).nonzero(as_tuple=True)[0]
    optimizer = optim.Adam([
        {'params': EnhancedNetwork.parameters(), 'lr': 0.0005},
        {'params': ReviseNetwork.parameters(), 'lr': 0.001}  
    ])

    # sof_output=F.softmax(output, dim=1)

    entropys = []
    for i, z in enumerate(F.softmax(output, dim=1)):
        entropy = -torch.sum(z * torch.log(z + 1e-9))  # 加上1e-9防止log(0)
        entropys.append(entropy)
    tensor = torch.tensor(entropys, dtype=torch.float32).cuda()
    mean = tensor.mean()

    uncertain_indices = []
    certain_indices = []
    for i, z in enumerate(F.softmax(output, dim=1)):
        if entropys[i] > mean * k:
            uncertain_indices.append(i)
        else:
            certain_indices.append(i)

    flag = 0
    if len(uncertain_indices)==0:
        flag=1

    # entropys=F.softmax(torch.Tensor(entropys))

    entropys=normalize2(torch.Tensor(entropys)).to(device)

    uncertain_indices = np.array(uncertain_indices)
    certain_indices = np.array(certain_indices)

    uncertain_indices_tarin = np.intersect1d(uncertain_indices, idx_train.cpu())
    certain_indices_tarin = np.intersect1d(certain_indices, idx_train.cpu())

    uncertain_indices_val = np.intersect1d(uncertain_indices, idx_test.cpu())
    certain_indices_val = np.intersect1d(certain_indices, idx_test.cpu())

    print("uncertain_indices num:",len(uncertain_indices))
    print("certain_indices num:", len(certain_indices))

    for epoch in range(epochs):
        optimizer.zero_grad()
        EnhancedNetwork.train()
        ReviseNetwork.train()
        output, ori_output, sof_output ,logsof_putput= model(features, adj)
        temp_output, temp_ori_output, temp_sof_output , temp_logsof_putput= model(features, adj)

        score_Enhanced = EnhancedNetwork(output)

        score_MLP = ReviseNetwork(features)

        losses_revised= F.nll_loss(score_MLP[idx_train], labels[idx_train])
        losses_reinforcement = F.nll_loss(score_Enhanced[certain_indices_tarin], labels[certain_indices_tarin])

        if math.isnan(losses_revised):
            losses_revised = 0
        if math.isnan(losses_reinforcement):
            losses_reinforcement = 0

        loss = losses_reinforcement + losses_revised
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            if epoch % 20 == 0:
                temp_output[uncertain_indices_val] = (score_MLP * (entropys.view(-1, 1)))[uncertain_indices_val] + \
                                                     (output * (1-entropys.view(-1, 1)))[uncertain_indices_val]
                temp_output[certain_indices_val] = (score_Enhanced * entropys.view(-1, 1))[certain_indices_val] + \
                                                   (output * (1 - entropys.view(-1, 1)))[certain_indices_val]

                finaloutput=temp_output
                acc_test = accuracy(finaloutput[idx_test], labels[idx_test])
                f1 = f1_test(finaloutput[idx_test], labels[idx_test])
                print(args.dataset,
                      "epoch=", epoch,
                      "k=", k,
                      "epoch=", epoch,
                      "accuracy= {:.2f}".format(100 * acc_test.item()),
                      "f1= {:.2f}".format(100 * f1.item()),
                      "loss= {:.4f}".format(loss))

                temp_output[uncertain_indices_val] = (score_MLP * 0.5)[uncertain_indices_val] + \
                                                     (output * 0.5)[uncertain_indices_val]
                temp_output[certain_indices_val] = (score_Enhanced * 0.5)[certain_indices_val] + \
                                                   (output * 0.5)[certain_indices_val]

                NoWF_output=temp_output
                NoWF_acc_test = accuracy(NoWF_output[idx_test], labels[idx_test])
                NoWF_f1 = f1_test(NoWF_output[idx_test], labels[idx_test])

                temp_output, _, _, _ = model(features, adj)
                temp_output[uncertain_indices_val] = (score_MLP * (entropys.view(-1, 1)))[uncertain_indices_val] + \
                                                     (output * (1-entropys.view(-1, 1)))[uncertain_indices_val]
                Nocer_output = temp_output
                Nocer_acc_test = accuracy(Nocer_output[idx_test], labels[idx_test])
                Nocer_f1 = f1_test(Nocer_output[idx_test], labels[idx_test])

                temp_output, _, _, _ = model(features, adj)
                temp_output[certain_indices_val] = (score_Enhanced * entropys.view(-1, 1))[certain_indices_val] + \
                                                   (output * (1 - entropys.view(-1, 1)))[certain_indices_val]
                Noun_output = temp_output
                Noun_acc_test = accuracy(Noun_output[idx_test], labels[idx_test])
                Noun_f1 = f1_test(Noun_output[idx_test], labels[idx_test])

    # # TSNE
    # prediction_test = F.log_softmax(output[idx_test], dim=1)
    # labels_test = labels[idx_test]
    # prediction_test = prediction_test.detach().cpu().numpy()
    # labels_test = labels_test.detach().cpu().numpy()
    # sio.savemat('{}_MvOGCN_linear_prediction_test.mat'.format(args.dataset), {'prediction_test': prediction_test})
    # sio.savemat('{}_MvOGCN_linear_labels_test.mat'.format(args.dataset), {'labels_test': labels_test})

    # loss_test = F.nll_loss(output[idx_test], labels[idx_test])
    # acc_test = accuracy(output[idx_test], labels[idx_test])
    # f1 = f1_test(output[idx_test], labels[idx_test])
    # print("Dataset:", args.dataset)
    # print("Test set results:",
    #       "accuracy= {:.2f}".format(100 * acc_test.item()),
    #       "f1= {:.2f}".format(100 * f1.item()))
    return 100 * acc_test.item(), 100 * f1.item(), flag,100 * NoWF_acc_test.item(), 100 * NoWF_f1.item(),100 * Nocer_acc_test.item(), 100 * Nocer_f1.item(),100 * Noun_acc_test.item(), 100 * Noun_f1.item()
