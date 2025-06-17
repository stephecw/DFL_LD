import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import os

lis = []
files = ["gap_200_100_10_0_1000000.txt", "gap_200_300_5_0_1000000.txt",]
legends = ["n=100, dim=10", "n=300, dim=5"]
for file in files:
    with open(file) as f:
        lines = f.readlines()
    lis.append(np.sort(np.array(list(map(float,lines[0].split(";"))))))

print(files)
# Calcul de la FRE
n = len(lis[0])
y = np.arange(1, n+1)

# Couleurs pour les approches
colors = ["red", "blue", "green", "black", "orange", "brown"]
for i, li in enumerate(lis):
    plt.step(li,np.arange(1, len(li)+1) / len(li) , color=colors[i%6], where='post', label=legends[i])
    # median = np.median(li)
    # max_value = np.max(li)
    # plt.axvline(median, color=colors[i%6], linestyle='--', ymax=np.interp(median, li, y))
    # plt.axvline(max_value, color=colors[i%6], linestyle=':')

plt.xlabel('Relative dual gap', fontsize=16)
plt.ylabel('Repartition', fontsize=16)
plt.title('200 instances', fontsize=18)
plt.grid(True)
plt.legend(fontsize=16)
plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
plt.show()

