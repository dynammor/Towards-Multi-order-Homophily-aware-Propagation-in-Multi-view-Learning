import matplotlib.pyplot as plt
import numpy as np
from matplotlib.font_manager import FontProperties


categories = ['100leaves', 'Animals', 'Flower17', 'GRAZ02', 'Youtube', 'NUS-WIDE', 'MNIST']
values1 = [91.44, 78.44, 64.85, 62.28, 64.83, 51.52 , 90.69]
values2 = [91.14, 78.13, 54.55, 58.9, 60.05, 47.26, 93.71]

plt.rcParams['font.family'] = 'Times New Roman'

bar_width = 0.35
index = np.arange(len(categories))


fig, ax = plt.subplots(figsize=(15, 10))


# ax.bar(index, values1, bar_width, label='MLP', color="#FFD9B7", edgecolor='black')
# ax.bar(index + bar_width, values2, bar_width, label='GCN', color="#D6E7B5", edgecolor='black')

ax.bar(index-0.04, values1, bar_width, label='MLP', color="#f6c6af", edgecolor='black')
ax.bar(index + bar_width+0.04, values2, bar_width, label='GCN', color="#b8b9d2", edgecolor='black')


# ax.set_title('Comparison of MLP and GCN', fontsize=17)
ax.set_xlabel('Dataset', fontsize=30)
ax.set_ylabel('Accuracy (%)', fontsize=30)

ax.set_xticks(index + bar_width / 2)
ax.set_xticklabels(categories, fontsize=25)
ax.tick_params(axis='y', labelsize=25)

ax.set_ylim(40, 100)


font_properties = FontProperties()
font_properties.set_size(24)
ax.legend(fontsize=24)

plt.tight_layout()

plt.savefig(f'C:\\Users\\asus\\Desktop\\论文\\contra\\mlp_gcn.pdf', format='pdf', dpi=600)

plt.show()