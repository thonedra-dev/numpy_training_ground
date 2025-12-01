import numpy as np
arr = np.array([
    [[1, 2, 3],
     [4, 5, 6]],
    
    [[7, 8, 9],
     [10, 11, 12]]
])

print("Original Array:\n", arr, "\n")

print("Sum of all elements in the array : ", np.sum(arr))
print("Largest of all elements in the array : ", np.max(arr))