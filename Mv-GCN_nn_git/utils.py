import torch
import torch.sparse

from sklearn.metrics import f1_score

import numpy as np

def compute_contribution_ratios_optimized(adj_matrix, gcn_output, pseudo_labels, num_class, device,neighbors_i):

    num_nodes = adj_matrix.shape[0]
    degrees = adj_matrix.sum(dim=1)


    sqrt_degrees = torch.sqrt(degrees).unsqueeze(1)
    norm_factors = 1 / (sqrt_degrees @ sqrt_degrees.T + 1e-10)

    ratios = {}


    label_matrix = (pseudo_labels.unsqueeze(0) == pseudo_labels.unsqueeze(1)).float()

    for v in range(num_nodes):


        v_neighbors = neighbors_i[v]

        if len(v_neighbors) == 0:
            ratios[v] = {
                'same_class_ratio': 0.0,
                'diff_class_ratio': 1.0
            }
            continue

        v_neighbors_tensor = torch.tensor(v_neighbors, dtype=torch.long)

        neighbor_contributions = gcn_output[v_neighbors_tensor] * norm_factors[v, v_neighbors_tensor].unsqueeze(-1)

        same_class_mask = label_matrix[v, v_neighbors_tensor].unsqueeze(-1)


        # same_class_contribution = (neighbor_contributions * same_class_mask).sum(dim=0)
        # diff_class_contribution = (neighbor_contributions * (1 - same_class_mask)).sum(dim=0)

        same_class_contribution = torch.norm(neighbor_contributions * same_class_mask, p=2)
        diff_class_contribution = torch.norm(neighbor_contributions * (1 - same_class_mask), p=2)

        total_contribution = same_class_contribution.sum() + diff_class_contribution.sum()


        same_class_ratio = same_class_contribution.sum().item() / total_contribution.item()

        # 存储结果
        ratios[v] = {
            'same_class_ratio': same_class_ratio,
            'diff_class_ratio': 1 - same_class_ratio
        }

    return ratios

def compute_contribution_ratios_optimized_two_hop(adj_matrix, gcn_output, pseudo_labels, num_class, device, neighbors):

    num_nodes = adj_matrix.shape[0]
    degrees = adj_matrix.sum(dim=1)
    sqrt_degrees = torch.sqrt(degrees).unsqueeze(1)
    norm_factors = 1 / (sqrt_degrees @ sqrt_degrees.T + 1e-10)

    ratios = {}

    label_matrix = (pseudo_labels.unsqueeze(0) == pseudo_labels.unsqueeze(1)).float()

    for v in range(num_nodes):

        v_neighbors = neighbors[v]


        two_hop_neighbors = set(v_neighbors)
        for neighbor in v_neighbors:
            two_hop_neighbors.update(neighbors[neighbor])
        two_hop_neighbors.discard(v)
        two_hop_neighbors = list(two_hop_neighbors)

        if len(two_hop_neighbors) == 0:
            ratios[v] = {
                'same_class_ratio': 0.0,
                'diff_class_ratio': 1.0
            }
            continue


        two_hop_neighbors_tensor = torch.tensor(two_hop_neighbors, dtype=torch.long)

        one_hop_contributions = gcn_output[v_neighbors] * norm_factors[v, v_neighbors].unsqueeze(-1)

        dz = degrees[two_hop_neighbors_tensor]
        dv = degrees[v]


        two_hop_norm_factors = (1 / dz) * norm_factors[v, two_hop_neighbors_tensor]

        two_hop_contributions = gcn_output[two_hop_neighbors_tensor] * two_hop_norm_factors.unsqueeze(-1)

        all_contributions = torch.cat([one_hop_contributions, two_hop_contributions], dim=0)

        same_class_mask = torch.cat(
            [label_matrix[v, v_neighbors], label_matrix[v, two_hop_neighbors_tensor]]).unsqueeze(-1)

        same_class_contribution = torch.norm(all_contributions * same_class_mask, p=2)
        diff_class_contribution = torch.norm(all_contributions * (1 - same_class_mask), p=2)

        total_contribution = same_class_contribution.sum() + diff_class_contribution.sum()

        same_class_ratio = same_class_contribution.sum().item() / total_contribution.item()

        ratios[v] = {
            'same_class_ratio': same_class_ratio,
            'diff_class_ratio': 1 - same_class_ratio
        }

    return ratios

import torch
from collections import deque

def compute_contribution_ratios_n_hop(adj_matrix, gcn_output, pseudo_labels, num_class, device, neighbors, max_hop=2, hop_weights=None):


    num_nodes = adj_matrix.shape[0]
    degrees = adj_matrix.sum(dim=1)


    sqrt_degrees = torch.sqrt(degrees).unsqueeze(1)
    norm_factors = 1 / (sqrt_degrees @ sqrt_degrees.T + 1e-10)


    ratios = {}


    label_matrix = (pseudo_labels.unsqueeze(0) == pseudo_labels.unsqueeze(1)).float()

    if hop_weights is None:
        hop_weights = [1.0 for _ in range(max_hop)]
    else:
        assert len(hop_weights) == max_hop, "hop_weights must = max_hop"

    for v in range(num_nodes):
        visited = set()
        queue = deque()
        for neighbor in neighbors[v]:
            if neighbor != v:
                queue.append((neighbor, 1))
                visited.add(neighbor)


        hop_neighbors = {hop: [] for hop in range(1, max_hop + 1)}
        while queue:
            node, hop = queue.popleft()
            if hop > max_hop:
                break
            hop_neighbors[hop].append(node)
            if hop < max_hop:
                for neighbor in neighbors[node]:
                    if neighbor != v and neighbor not in visited:
                        queue.append((neighbor, hop + 1))
                        visited.add(neighbor)


        all_hop_neighbors = []
        all_hop_weights = []
        for hop in range(1, max_hop + 1):
            all_hop_neighbors.extend(hop_neighbors[hop])
            all_hop_weights.extend([hop_weights[hop - 1]] * len(hop_neighbors[hop]))

        if len(all_hop_neighbors) == 0:

            ratios[v] = {
                'same_class_ratio': 0.0,
                'diff_class_ratio': 0.0
            }
            continue

        all_hop_neighbors_tensor = torch.tensor(all_hop_neighbors, dtype=torch.long, device=device)
        all_hop_weights_tensor = torch.tensor(all_hop_weights, dtype=torch.float, device=device).unsqueeze(-1)

        dz = degrees[all_hop_neighbors_tensor]
        # norm_factors[v, z] = 1 / (sqrt(d_v) * sqrt(d_z))
        two_hop_norm_factors = (1 / dz) * norm_factors[v, all_hop_neighbors_tensor]

        normalized_factors = two_hop_norm_factors.unsqueeze(-1) * all_hop_weights_tensor

        contributions = gcn_output[all_hop_neighbors_tensor] * normalized_factors

        same_class_mask = label_matrix[v, all_hop_neighbors_tensor].unsqueeze(-1)

        same_class_contribution = torch.norm(contributions * same_class_mask, p=2)
        diff_class_contribution = torch.norm(contributions * (1 - same_class_mask), p=2)

        total_contribution = same_class_contribution + diff_class_contribution
        if total_contribution.item() == 0:
            same_class_ratio = 0.0
        else:
            same_class_ratio = same_class_contribution.item() / (total_contribution.item() + 1e-10)

        ratios[v] = {
            'same_class_ratio': same_class_ratio,
            'diff_class_ratio': 1 - same_class_ratio
        }

    return ratios


def compute_contribution_ratios_two_hop_only(adj_matrix, gcn_output, pseudo_labels, num_class, device, neighbors):

    num_nodes = adj_matrix.shape[0]
    degrees = adj_matrix.sum(dim=1)

    sqrt_degrees = torch.sqrt(degrees).unsqueeze(1)
    norm_factors = 1 / (sqrt_degrees @ sqrt_degrees.T + 1e-10)

    ratios = {}


    label_matrix = (pseudo_labels.unsqueeze(0) == pseudo_labels.unsqueeze(1)).float()

    for v in range(num_nodes):
        v_neighbors = neighbors[v]

        two_hop_neighbors = set(v_neighbors)
        for neighbor in v_neighbors:
            two_hop_neighbors.update(neighbors[neighbor])
        two_hop_neighbors.discard(v)
        two_hop_neighbors = list(two_hop_neighbors)

        if len(two_hop_neighbors) == 0:
            ratios[v] = {
                'same_class_ratio': 0.0,
                'diff_class_ratio': 1.0
            }
            continue


        two_hop_neighbors_tensor = torch.tensor(two_hop_neighbors, dtype=torch.long)


        dz = degrees[two_hop_neighbors_tensor]
        dv = degrees[v]

        two_hop_norm_factors = (1 / dz) * norm_factors[v, two_hop_neighbors_tensor]

        two_hop_contributions = gcn_output[two_hop_neighbors_tensor] * two_hop_norm_factors.unsqueeze(-1)

        same_class_mask = label_matrix[v, two_hop_neighbors_tensor].unsqueeze(-1)

        same_class_contribution = torch.norm(two_hop_contributions * same_class_mask, p=2)
        diff_class_contribution = torch.norm(two_hop_contributions * (1 - same_class_mask), p=2)

        total_contribution = same_class_contribution.sum() + diff_class_contribution.sum()

        same_class_ratio = same_class_contribution.sum().item() / total_contribution.item()

        ratios[v] = {
            'same_class_ratio': same_class_ratio,
            'diff_class_ratio': 1 - same_class_ratio
        }

    return ratios

def softmax(logits):
    exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
    return exp_logits / np.sum(exp_logits, axis=1, keepdims=True)


def convert_to_classes(output):

    probs = softmax(output)

    predicted_classes = np.argmax(probs, axis=1)
    return predicted_classes

def get_neighbors(adj_matrix, node):

    neighbors = np.where(adj_matrix[node] > 0)[0]
    return neighbors

def calculate_probability(C, e, m, p_same, p_diff, n):

    term1 = m * (p_same - p_diff)
    term2 = n * (p_diff * (C - 1) / (C - 2) - 1 / (C - 2))
    exp_term = np.exp(-e * (term1 + term2))
    P_class_1 = 1 / (1 + (C - 1) * exp_term)

    return P_class_1


def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)

def f1_test(output, labels):
    preds = output.max(1)[1].type_as(labels)
    # correct = preds.eq(labels).double()
    f1 = f1_score(preds.detach().cpu().numpy(), labels.detach().cpu().numpy(), average='macro')
    return f1




def sparse_mx_to_torch_sparse_tensor(sparse_mx, device=None):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    tensor = torch.sparse.FloatTensor(indices, values, shape)
    if device is not None:
        tensor = tensor.to(device)
    return tensor




# def draw_plt(output_, labels):
#     output_ = output_.detach().cpu().numpy()
#     labels = labels.detach().cpu().numpy()
#     X_tsne = manifold.TSNE(n_components=2, learning_rate=100, random_state=42).fit_transform(output_)
#     plt.figure(figsize=(8, 6))
#     # plt.title('Dataset : ' + dataset_name + '   (Label rate : 20 nodes per class)')
#
#     # for i in index:
#     #     plt.scatter(X_tsne[i, 0], X_tsne[i, 1], c='none', marker='o', edgecolors='black', s=30)
#
#     scatter = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=labels, s=8)
#     handles, _ = scatter.legend_elements(prop='colors')
#     labels = ['1', '2', '3', '4', '5', '6', '7', '8', '9']
#     plt.legend(handles, labels, loc='upper right')
#     # plt.colorbar(ticks=range(5))
#     # plt.savefig('./result/tsne/cnae.svg')
#     plt.show()

