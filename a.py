import numpy as np

# (weight, bias, feature_value) Computation

X = np.array([
    [1, 2],
    [3, 4],
    [5, 6]
])

w = np.array([2,4])

b = 5

Z = np.dot(X,w) + b

print(Z)