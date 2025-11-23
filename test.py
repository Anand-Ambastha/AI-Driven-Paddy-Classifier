import pandas as pd
import numpy as np
f = np.load('y.npy')
f2 = np.load('X.npy')
print(f2.shape)
print(np.unique(f))


f3 = np.load('y_test.npy')
f4 = np.load('X_test.npy')
print(f4.shape)
print(np.unique(f3))