import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from matplotlib.ticker import MaxNLocator



matplotlib.rcParams['font.family'] = 'Times New Roman'
matplotlib.rcParams['font.size'] =5

# data = 'scene15'
categories = ['w/o All', 'w/o U', 'w/o WF', 'Ours']

# data = '100leaves'
# values = [91.14, 92.95, 92.21, 95.33]
# errors = [0.92, 0.20, 0.66, 0.24]

# data = 'animals'
# values = [78.13, 79.66, 77.03, 83.53]
# errors = [0.54, 0.31, 0.74, 0.70]
# #
data = 'flower17'
values = [54.55, 56.97, 61.27, 68.00]
errors = [3.55, 3.73, 1.53, 1.00]
#
# data = 'GRAZ02'
# values = [55.45, 56.40, 61.40, 66.06]
# errors = [1.50, 1.44, 2.06, 1.32]
#
# data = 'MNIST10k'
# values = [93.71, 93.98, 93.68, 93.94]
# errors = [0.18, 0.04, 0.08, 0.04]
#
# data ='MSRC-v1'
# values = [84.48, 85.36, 82.36, 91.36]
# errors = [3.24, 1.32, 0.66, 2.13]
#
#
# data = 'scene15'
# values = [69.9298,72.07,74.399, 79.92]
# errors = [ 3.4609,0.860,0.777, 0.126]

# data ='Out_Scene'
# values = [78.79, 79.40, 79.88, 82.97]
# errors = [0.42, 0.27, 0.30, 0.49]


x_pos = np.arange(len(categories))

colors = ['#f6c6af', '#b8b9d2', '#b5d4be', '#afd4e3']

plt.figure(figsize=(6.5, 6))
plt.bar(
    x_pos,
    values,
    yerr=errors,
    capsize=5,
    width=0.50,
    color=colors,
    edgecolor='black',
    error_kw={'elinewidth': 2, 'ecolor': 'black'},
    zorder=2
)



plt.ylabel('Accuracy (%)', fontsize=30)
plt.xticks(x_pos, categories, fontsize=30)
plt.yticks(fontsize=30)


ax = plt.gca()
# ax.spines['right'].set_visible(False)
# ax.spines['top'].set_visible(False)


plt.ylim(50, 70)

ax = plt.gca()
ax.yaxis.set_major_locator(MaxNLocator(integer=True, nbins=5))
ax.grid(axis='both', linestyle='-', color='gray', alpha=0.3, zorder=1)

plt.tight_layout()


plt.savefig(f'C:\\Users\\asus\\Desktop\\paper\\wo\\{data}_2.pdf', format='pdf', dpi=600)

plt.show()
