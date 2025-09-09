import os

import numpy as np
import scipy.io as scio
from sklearn.neighbors import kneighbors_graph
import torch
import scipy.sparse as sp


### Data normalization
def normalization(data):
    maxVal = torch.max(data)
    minVal = torch.min(data)
    data = (data - minVal) // (maxVal - minVal)
    return data


### Data standardization
def standardization(data):
    rowSum = torch.sqrt(torch.sum(data ** 2, 1))
    repMat = rowSum.repeat((data.shape[1], 1)) + 1e-10
    data = torch.div(data, repMat.t())
    return data


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    # return torch.sparse.FloatTensor(indices, values, shape)
    return torch.sparse_coo_tensor(indices, values, shape, dtype=torch.float)


def construct_laplacian(adj):
    """
        construct the Laplacian matrix
    :param adj: original Laplacian matrix  <class 'scipy.sparse.csr.csr_matrix'>
    :return:
    """
    adj_ = sp.coo_matrix(adj)
    # adj_ = sp.eye(adj.shape[0]) + adj
    rowsum = np.array(adj_.sum(1))  # <class 'numpy.ndarray'> (n_samples, 1)
    # print("mean_degree:", rowsum.mean())
    degree_mat_inv_sqrt = sp.diags(np.power(rowsum, -0.5).flatten())  # degree matrix
    # <class 'scipy.sparse.coo.coo_matrix'>  (n_samples, n_samples)
    adj_wave = adj_.dot(degree_mat_inv_sqrt).transpose().dot(degree_mat_inv_sqrt)
    lp = sp.eye(adj.shape[0]) - adj_wave
    # lp = adj_wave
    return lp

def normalize2(arr):
    tensor_min = torch.min(arr)
    tensor_max = torch.max(arr)

    normalized_arr = (arr - tensor_min) / (tensor_max - tensor_min)
    return normalized_arr
def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1), dtype=np.float32)
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def LoadMatData(datasets, k, path="D:/Lab/dataset/multiview datasets/"):
    print("loading {} data...".format(datasets))
    data = scio.loadmat("{}{}.mat".format(path, datasets))
    x = data["X"]
    adj = []
    adj_noself = []
    features = []
    nfeats = []
    for i in range(x.shape[1]):
        features.append(x[0, i].astype('float32'))
        nfeats.append(len(features[i][1]))
        temp = kneighbors_graph(features[i], k)
        temp = sp.coo_matrix(temp)
        temp = temp + temp.T.multiply(temp.T > temp) - temp.multiply(temp.T > temp)
        temp = normalize(temp + sp.eye(temp.shape[0]))
        temp_noself=normalize(temp)
        temp = sparse_mx_to_torch_sparse_tensor(temp)
        temp_noself = sparse_mx_to_torch_sparse_tensor(temp_noself)
        adj.append(temp)
        adj_noself.append(temp_noself)
    # for i in range(x.shape[1]):
    #     features[i] = np.transpose(features[i])
    labels = data["Y"].reshape(-1, ).astype('int64')
    labels = labels - min(set(labels))
    num_class = len(set(np.array(labels)))
    labels = torch.from_numpy(labels)
    return adj, features, labels, nfeats, len(nfeats), num_class,adj_noself

#
# def dataloader(dataset_name, k_value, load_from_file=True):
#     data_dir = os.path.join("./training_data", dataset_name)
#     exist_k = os.path.exists(os.path.join(data_dir, "data_" + str(k_value) + ".pth"))
#     if load_from_file and os.path.exists(data_dir) and exist_k:
#         print("Loading Existing adjacency matrix.")
#         load_data = torch.load(os.path.join(data_dir, "data_" + str(k_value) + ".pth"))
#         load_adjs = torch.load(os.path.join(data_dir, "adj_" + str(k_value) + ".pth"))
#         features = load_data["features"]
#         if dataset_name == "Reuters":
#             for idx, feature in enumerate(features[0]):
#                 features[0][idx] = feature.todense()
#         gnd = load_data["gnd"]
#         adjs = load_adjs["adjs"].to_dense()
#         adj_hats = load_adjs["adj_hats"].to_dense()
#
#     else:
#         print("Generating adjacency matrix.")
#         features, gnd = loadMatData(os.path.join("D:\dataset\multiview datasets\\", dataset_name))
#         features = feature_normalization(features)
#         adjs, adj_hats = load_adjacency_multiview(features, k=k_value)
#         adjs = torch.from_numpy(adjs)
#         adj_hats = torch.from_numpy(adj_hats)
#         if not os.path.exists(data_dir):
#             os.mkdir(data_dir)
#         save_data = {'features': features, 'gnd': gnd}
#         save_adjs = {'adjs': adjs.to_sparse(), 'adj_hats': adj_hats.to_sparse()}
#
#         torch.save(save_data, os.path.join(data_dir, "data_" + str(k_value) + ".pth"))
#         torch.save(save_adjs, os.path.join(data_dir, "adj_" + str(k_value) + ".pth"))
#
#         if dataset_name == "Reuters":
#             for idx, feature in enumerate(features[0]):
#                 features[0][idx] = feature.todense()
#
#     data_split = torch.load(os.path.join(data_dir, "data_split.pth"))
#     p_labeled = data_split["training"]
#     p_unlabeled = data_split["testing"]
#
#     return features, gnd, p_labeled, p_unlabeled, adjs, adj_hats
#

# def loadMatData(data_name):
#     data = scio.loadmat(data_name)
#     features = data['X']  # .dtype = 'float32'
#     gnd = data['Y']
#     gnd = gnd.flatten()
#     gnd = gnd - min(set(gnd))
#     return features, gnd


def feature_normalization(features, normalization_type='normalize'):
    for idx, fea in enumerate(features[0]):
        if normalization_type == 'normalize':
            features[0][idx] = normalize(fea)
        else:
            print("Please enter a correct normalization type!")
    return features


def count_each_class_num(gnd):
    '''
    Count the number of samples in each class
    '''
    count_dict = {}
    for label in gnd:
        if label in count_dict.keys():
            count_dict[label] += 1
        else:
            count_dict[label] = 1
    return count_dict


def generate_permutation(gnd, args):
    '''
    Generate permutation for training, validating and testing data.
    '''
    gnd = gnd.cpu().numpy()
    gnd = np.array(gnd)
    N = gnd.shape[0]
    each_class_num = count_each_class_num(gnd)
    training_each_class_num = {}  ## number of labeled samples for each class
    # valid_each_class_num = {}
    # test_each_class_num = {}
    for label in each_class_num.keys():
        if args.data_split_mode == "Ratio":
            train_ratio = args.train_ratio
            # valid_ratio = args.valid_ratio
            test_ratio = 1 - args.train_ratio
            training_each_class_num[label] = max(round(each_class_num[label] * train_ratio), 1)  # min is 1
            # valid_num = max(round(N * valid_ratio), 1)  # min is 1
            test_num = max(round(N * test_ratio), 1)  # min is 1
        else:
            training_each_class_num[label] = args.num_train_per_class
            valid_num = args.num_val
            test_num = args.num_test

    # index of labeled and unlabeled samples
    train_mask = torch.from_numpy(np.full((N), False))
    # valid_mask = torch.from_numpy(np.full((N), False))
    test_mask = torch.from_numpy(np.full((N), False))

    train_idx = []
    valid_idx = []
    test_idx = []

    # shuffle the data
    data_idx = np.random.permutation(range(N))

    # Get training data
    for idx in data_idx:
        label = gnd[idx]
        if (training_each_class_num[label] > 0):
            training_each_class_num[label] -= 1
            train_mask[idx] = True
            train_idx.append(idx)
    for idx in data_idx:
        if train_mask[idx] == True:
            continue
        # if (valid_num > 0):
        #     valid_num -= 1
        #     valid_mask[idx] = True
        #     valid_idx.append(idx)
        elif (test_num > 0):
            test_num -= 1
            test_mask[idx] = True
            test_idx.append(idx)
    return torch.from_numpy(np.array(train_idx)).long(), torch.from_numpy(
        np.array(test_idx)).long()
