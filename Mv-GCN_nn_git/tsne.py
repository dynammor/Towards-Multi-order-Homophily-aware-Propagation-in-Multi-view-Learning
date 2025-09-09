import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import numpy as np


def draw_plt(dataset_name, output_, labels):
    X_tsne = TSNE(n_components=2, learning_rate=100, random_state=42).fit_transform(output_)
    plt.figure(figsize=(8, 6))
    # plt.title('Dataset : ' + dataset_name + '   (Label rate : 20 nodes per class)')
    scatter = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=labels, s=8,cmap='rainbow')
    handles, _ = scatter.legend_elements(prop='colors')
    # plt.axis('off')
    labels = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10']
    # plt.legend(handles, labels, loc='upper right')

    plt.axis('off')
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['bottom'].set_visible(False)
    plt.gca().spines['left'].set_visible(False)
    # plt.colorbar(ticks=range(3))
    plt.savefig('C:\\Users\\asus\\Desktop\\paper\\tsne\\'+dataset_name + '.pdf', transparent=True,format='pdf')
    plt.show()


def permute_adj(affinity, labels, n_class):
    new_ind = []
    for i in range(n_class):
        ind = np.where(labels == i)[0].tolist()
        new_ind += ind
    return affinity[new_ind, :][:, new_ind]
# def visulization(P):