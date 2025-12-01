import numpy as np

arr = np.array([
    [[ 1,  3,  5,  7],
     [ 9, 11, 13, 15]],

    [[ 2,  4,  6,  8],
     [10, 12, 14, 16]],

    [[17, 19, 21, 23],
     [25, 27, 29, 31]]
])

print(arr.shape)

print("This is the sum of all elements in the whole array", arr.sum())
print("This is the sum of all elements in the whole array", arr.max())
print("This is the sum of all elements in the whole array", arr.min())

# Starting from below, we will deal with the axis = 0 | 1 | 2, things.