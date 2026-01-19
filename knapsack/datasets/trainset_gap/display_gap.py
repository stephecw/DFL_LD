import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import os

lis = []
files = []

for file in os.listdir("./"):
    if os.path.isfile(file) and file[-4:] == ".txt":
        files.append(file[:-4])
        with open(file) as f:
            lines = f.readlines()
        lis.append(np.sort(np.array(list(map(float,lines[0].split(";"))))))

print(files)
# Compute the ECDF
n = len(lis[0])
y = np.arange(1, n+1)

# Colors for each approach
colors = ["red", "blue", "green", "black", "orange", "brown"]
for i, li in enumerate(lis):
    plt.step(li,np.arange(1, len(li)+1) / len(li) , color=colors[i%6], where='post', label=files[i][4:])
    # median = np.median(li)
    # max_value = np.max(li)
    # plt.axvline(median, color=colors[i%6], linestyle='--', ymax=np.interp(median, li, y))
    # plt.axvline(max_value, color=colors[i%6], linestyle=':')

plt.xlabel('Relative gap')
plt.ylabel('Distribution')
plt.title('n=30, dim=5')
plt.legend()
plt.grid(True)
plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
plt.show()
