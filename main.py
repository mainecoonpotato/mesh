import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

wf = [[0., 0.]]
n_r = 5
n_s = 4

r0 = 150 / np.sqrt(n_r)
s0 = 2 * np.pi / n_s

for ni_r in range(n_r):
    ri = r0 * np.sqrt(ni_r + 1)
    for ni_s in range(n_s):
        wf.append([ri * np.cos(s0 * ni_s), ri * np.sin(s0 * ni_s)])
        print(ri, ni_s)

wf = np.array(wf)

plt.scatter(wf[:, 0], wf[:, 1])
plt.show()

