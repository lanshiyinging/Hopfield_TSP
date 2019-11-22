import numpy as np

a = np.zeros((3, 3))
n = 0
for i in range(3):
    for j in range(3):
        a[i, j] = n
        n += 1
print(a)
print(np.sum(a, axis=0))