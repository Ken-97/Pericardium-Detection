import numpy as np
import png
a = np.zeros((3,4))
b = 1
for i in np.arange(3):
    for j in np.arange(4):
        a[i,j] = b
        b = b + 1

c = a > 8
a[c] = 0
print a
