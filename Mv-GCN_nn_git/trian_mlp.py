from __future__ import division
from __future__ import print_function

import math
import time

import numpy as np
import scipy.sparse as sp
# import sio
from sklearn.neighbors import kneighbors_graph
from scipy.io import savemat
from DataLoader import LoadMatData, generate_permutation, normalize, normalize2
import torch.nn.functional as F
import torch.optim as optim
from MvMLP import MvMLP
from MvGCN import MvGCN, MvGCN_onehop
from tsne import draw_plt

from utils import accuracy, f1_test, get_neighbors, convert_to_classes, calculate_probability, \
    sparse_mx_to_torch_sparse_tensor, compute_contribution_ratios_optimized, \
    compute_contribution_ratios_optimized_two_hop
from tqdm import tqdm
import os
import torch


def train(adj, features, labels, nfeats, num_view, num_class, args, device,sage,layer_num):
    idx_train, idx_test = generate_permutation(labels, args)
    # model = MvMLP(nfeats, num_class)
    if layer_num==1:
        model =MvGCN_onehop(nfeats, num_class)
    else:
        model = MvGCN(nfeats, num_class)
    # model = MvFGCN(nfeats, num_class)
    # model = MvACMGCN(nfeats, num_class)
    total_para = sum(x.numel() for x in model.parameters())
    # print("Total number of paramerters in networks is {}  ".format(total_para / 1e6))
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    if sage==0:
        for i in range(num_view):
            exec("features_{}= torch.from_numpy(features[{}]/1.0).float().to(device)".format(i, i))
            exec("features[{}]= torch.Tensor(features[{}] / 1.0).to(device)".format(i, i))
            exec("features[{}] = F.normalize(features[{}])".format(i, i))
            exec("adj[{}]=adj[{}].to_dense().float().to(device)".format(i, i))
    else:
        for i in range(num_view):
            exec("adj[{}]=adj[{}].to_dense().float().to(device)".format(i, i))

    if args.cuda:
        model.to(device)
        labels = labels.to(device)
        idx_train = idx_train.to(device) # [:10]
        idx_test = idx_test.to(device)
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
            output, _, _, _ ,_= model(features, adj)
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
                output, _, _, _ ,_= model(features, adj)
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

    # print('total_time:', time.time() - t1)
    torch.save({
        'x_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, f'./checkpoints/best_{args.dataset}.pth')
    return model, features, labels, adj, idx_test, output, idx_train

def test(args, model, features, labels, adj, idx_test):
    model.eval()
    output, ori_output, sof_output, logsof_putput ,_= model(features, adj)

    # # TSNE
    # prediction_test = F.log_softmax(output[idx_test], dim=1)
    # labels_test = labels[idx_test]
    # prediction_test = prediction_test.detach().cpu().numpy()
    # labels_test = labels_test.detach().cpu().numpy()
    # sio.savemat('{}_MvOGCN_linear_prediction_test.mat'.format(args.dataset), {'prediction_test': prediction_test})
    # sio.savemat('{}_MvOGCN_linear_labels_test.mat'.format(args.dataset), {'labels_test': labels_test})

    # loss_test = F.nll_loss(output[idx_test], labels[idx_test])

    acc_test = accuracy(output[idx_test], labels[idx_test])
    f1 = f1_test(output[idx_test], labels[idx_test])
    return 100 * acc_test.item(), 100 * f1.item()


def Ftmodel(args, model, features, labels, adj, idx_test, EnhancedNetwork, ReviseNetwork, num_class, device, idx_train,
            k, epoch_2,nfeats,alpha,neighbors,model_noself_twoloop,model_noself_oneloop):

    # checkpoint = torch.load(f'./checkpoints/best_{args.dataset}.pth')
    # # model.load_state_dict(checkpoint['x_state_dict'])
    # # model_noself_oneloop.load_state_dict(checkpoint['x_state_dict'])
    # # model_noself_twoloop.load_state_dict(checkpoint['x_state_dict'])
    tempa = alpha * 0.1
    tempk=k*0.1

    acc_300=[]
    acc_500=[]
    # model_weights = model.state_dict()
    # for name, param in model_weights.items():
    #     print(f"Layer: {name} | Shape: {param.shape}")

    model.eval()
    model_noself_oneloop.eval()
    model_noself_twoloop.eval()

    output, ori_output, sof_output, logsof_putput, GCN_outputs = model(features, adj)

    predicted_classes = torch.argmax(output, dim=1)
    incorrect_indices = (predicted_classes != labels).nonzero(as_tuple=True)[0]

    optimizer = optim.Adam([
        {'params': model.parameters(), 'lr': 0.001},
        {'params': ReviseNetwork.parameters(), 'lr': 0.001}
    ])

    uncertain_indices = []
    certain_indices = []
    certain_indices_final= []
    contributions_onehop = []
    contributions_twohop = []

    mult_hs_1=[]
    mult_hs_2 = []

    for i, (a, feat) in enumerate(zip(adj, features)):
        #gcn_weight=torch.matmul(model_weights[f'GCNs.{i}.gc_in.weight'],model_weights[f'GCNs.{i}.gc_out.weight'])
        gcn_weight=0
        contribution_onehop =  compute_contribution_ratios_optimized(a, feat, predicted_classes,gcn_weight,num_class,neighbors[i])
        contribution_twohop = compute_contribution_ratios_optimized_two_hop(a, feat, predicted_classes, num_class, device, neighbors[i])
        contributions_onehop.append(contribution_onehop)
        contributions_twohop.append(contribution_twohop)
        hs_1 = []
        hs_2 = []
        for j, z in enumerate(F.softmax(output, dim=1)):
            h_1=contribution_onehop[j]['diff_class_ratio']
            h_2 = contribution_twohop[j]['diff_class_ratio']
            hs_1.append(h_1)
            hs_2.append(h_2)
        mult_hs_1.append(hs_1)
        mult_hs_2.append(hs_2)

    means_1 = [sum(x) / len(x) for x in zip(*mult_hs_1)]
    means_2 = [sum(x) / len(x) for x in zip(*mult_hs_2)]

    indices_two = [i for i in range(len(means_1)) if
                   (means_1[i] + means_2[i] == 0) or (means_1[i] / (means_1[i] + means_2[i]) > tempa)]
    indices_one = [i for i in range(len(means_1)) if
                   (means_1[i] + means_2[i] != 0) and (means_1[i] / (means_1[i] + means_2[i]) <= tempa)]

    for i, z in enumerate(F.softmax(output, dim=1)):
        # if entropys[i] >math.log(num_class) *k:
        if means_1[i] > tempk:
            uncertain_indices.append(i)
        else :
            certain_indices_final.append(i)
        if means_1[i] < 0.7:
            certain_indices.append(i)

    correct_predictions = [i for i in uncertain_indices if i in incorrect_indices]

    if  len(uncertain_indices)!=0:
        correct_prediction_probability = len(correct_predictions) / len(uncertain_indices)
        print(f"错误比例: {correct_prediction_probability}")
    else:
        correct_prediction_probability=0
    correct_predictions_cer = [i for i in certain_indices if i in incorrect_indices]
    if len(certain_indices) != 0:
        correct_prediction_probability_cer = len(correct_predictions_cer) / len(certain_indices)
        print(f"cer中错误比例: {correct_prediction_probability_cer}")
    else:
        correct_prediction_probability_cer=0

    flag = 0
    if len(certain_indices) == 0:
        flag = 1

    means_1 =torch.Tensor(means_1).to(device)
    # means_1 = normalize2(means_1).to(device)
    means_2 =torch.Tensor(means_2).to(device)
    # means_2 = normalize2(means_2).to(device)

    uncer_indices_two =  np.array(list(set(indices_two) & set(uncertain_indices)))
    uncer_indices_one =  np.array(list(set(indices_one) & set(uncertain_indices)))

    uncertain_indices = np.array(uncertain_indices)
    certain_indices = np.array(certain_indices)

    uncertain_indices_tarin = np.intersect1d(uncertain_indices, idx_train.cpu())
    certain_indices_tarin = np.intersect1d(certain_indices, idx_train.cpu())

    uncertain_indices_val = np.intersect1d(uncertain_indices, idx_test.cpu())
    certain_indices_val = np.intersect1d(certain_indices, idx_test.cpu())

    uncertain_indices_val_one =  np.array(list(set(uncertain_indices_val) & set(uncer_indices_one)))
    uncertain_indices_val_two =  np.array(list(set(uncertain_indices_val) & set(uncer_indices_two)))

    print("uncertain_indices num:", len(uncertain_indices))
    print("certain_indices num:", len(certain_indices))
    pa_loss = []
    pa_f1 = []
    pa_acc = []
    pa_epoch = []
    for epoch in range(epoch_2):
        pa_epoch.append(epoch)
        optimizer.zero_grad()
        EnhancedNetwork.train()
        ReviseNetwork.train()

        output_one, ori_output, sof_output, logsof_putput,_ = model_noself_oneloop(features, adj)
        output_two, ori_output, sof_output, logsof_putput, _ = model_noself_twoloop(features, adj)

        temp_output, temp_ori_output, temp_sof_output, temp_logsof_putput,_= model(features, adj)
        temp_output = F.log_softmax(temp_output, dim=1)

        model.train()

        score_Enhanced, ori_output, sof_output, logsof_putput,_ = model(features, adj)
        score_Enhanced = F.log_softmax(score_Enhanced, dim=1)

        # score_Enhanced = EnhancedNetwork(output)
        # score_Enhanced = F.log_softmax(score_Enhanced, dim=1)

        score_MLP = ReviseNetwork(features)
        score_MLP = F.log_softmax(score_MLP, dim=1)

        losses_revised = F.nll_loss(score_MLP[idx_train], labels[idx_train])
        losses_reinforcement = F.nll_loss(score_Enhanced[certain_indices_tarin], labels[certain_indices_tarin])
        # losses_reinforcement = F.nll_loss(score_Enhanced[certain_indices_tarin], labels[certain_indices_tarin])

        if math.isnan(losses_revised):
            losses_revised = 0
        if math.isnan(losses_reinforcement):
            losses_reinforcement = 0

        loss = losses_reinforcement + losses_revised



        loss.backward()
        optimizer.step()
        pa_loss.append(loss.item())

        with torch.no_grad():
            ReviseNetwork.eval()
            model.eval()
            EnhancedNetwork.eval()

            score_MLP = ReviseNetwork(features)
            score_MLP = F.log_softmax(score_MLP, dim=1)
            score_Enhanced, ori_output, sof_output, logsof_putput, _ = model(features, adj)
            score_Enhanced = F.log_softmax(score_Enhanced, dim=1)

            if epoch % 1 == 0:
                temp_output[uncertain_indices_val_one] = (score_MLP * (means_1.view(-1, 1)))[uncertain_indices_val_one] + \
                                                      (output_one * (1 - means_1.view(-1, 1)))[uncertain_indices_val_one]
                temp_output[uncertain_indices_val_two] = (score_MLP * (means_2.view(-1, 1)))[uncertain_indices_val_two] + \
                                                     (output_two * (1 - means_2.view(-1, 1)))[uncertain_indices_val_two]
                temp_output[certain_indices_final] = (score_Enhanced * (means_1.view(-1, 1)))[certain_indices_final] + \
                                                    (output * (1 - means_1.view(-1, 1)))[certain_indices_final]

                #temp_output[uncertain_indices_val] = (score_MLP)[uncertain_indices_val]
                #temp_output[certain_indices_final] = (score_Enhanced )[certain_indices_final] 

                finaloutput = temp_output
                acc_test = accuracy(finaloutput[idx_test], labels[idx_test])

                acc_test2 = accuracy(score_MLP[idx_test], labels[idx_test])
                if epoch  == 300:
                    acc_300=100 * acc_test.item()
                elif epoch  == 500:
                    acc_500 = 100 * acc_test.item()
                f1 = f1_test(finaloutput[idx_test], labels[idx_test])
                pa_acc.append(100*acc_test.item())
                pa_f1.append(100*f1.item())

                # print(args.dataset,
                #       "epoch=", epoch,
                #       "k=", k,
                #       "a=", alpha,
                #       "epoch=", epoch,
                #       "accuracy= {:.2f}".format(100 * acc_test.item()),
                #       "accuracy2= {:.2f}".format(100 * acc_test2.item()),
                #       "f1= {:.2f}".format(100 * f1.item()),
                #       "loss= {:.4f}".format(loss))

                temp_output, _, _, _,_ = model(features, adj)
                temp_output[uncertain_indices_val] = (score_MLP)[uncertain_indices_val] 
                temp_output[certain_indices_final] = (score_Enhanced)[certain_indices_final] 

                NoWF_output = temp_output
                NoWF_acc_test = accuracy(NoWF_output[idx_test], labels[idx_test])
                NoWF_f1 = f1_test(NoWF_output[idx_test], labels[idx_test])

                temp_output, _, _, _ ,_= model(features, adj)
                temp_output[uncertain_indices_val] = (score_MLP * ( means_1.view(-1, 1)))[uncertain_indices_val] + \
                                                     (output * (1 -means_1.view(-1, 1)))[uncertain_indices_val]
                Nocer_output = temp_output
                Nocer_acc_test = accuracy(Nocer_output[idx_test], labels[idx_test])
                Nocer_f1 = f1_test(Nocer_output[idx_test], labels[idx_test])

                temp_output, _, _, _ ,_= model(features, adj)
                temp_output[certain_indices_val] = (score_Enhanced * means_1.view(-1, 1))[certain_indices_val] + \
                                                   (output * (1 - means_1.view(-1, 1)))[certain_indices_val]
                Noun_output = temp_output
                Noun_acc_test = accuracy(Noun_output[idx_test], labels[idx_test])
                Noun_f1 = f1_test(Noun_output[idx_test], labels[idx_test])

    # draw_plt(args.dataset, finaloutput.detach().cpu().numpy(), labels.detach().cpu().numpy())

    data_dict = {
        'pa_f1': np.array(pa_f1),
        'pa_epoch': np.array(pa_epoch),
        'pa_acc': np.array(pa_acc),
        'pa_loss':np.array(pa_loss)
    }
    # 保存到 .mat 文件
    savemat(f'./results/mat/{args.dataset}_loss_{epoch_2}.mat', data_dict)

    # output = output.cpu().detach().numpy()
    # labels=labels.cpu().detach().numpy()
    # predicted_classes = torch.argmax(output, dim=1)
    # plot_tsne_with_ellipses(output, predicted_classes, num_class)
    #
    # predicted_classes = torch.argmax(finaloutput, dim=1)
    # plot_tsne_with_ellipses(finaloutput, predicted_classes, num_class)

    # TSNE
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
    return 100 * acc_test.item(), 100 * f1.item(), flag, 100 * NoWF_acc_test.item(), 100 * NoWF_f1.item(), 100 * Nocer_acc_test.item(), 100 * Nocer_f1.item(), 100 * Noun_acc_test.item(), 100 * Noun_f1.item(),correct_prediction_probability,acc_300,acc_500
