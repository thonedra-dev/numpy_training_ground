import numpy as np

arr = np.array([
    [[ 32,  3,  51,  7],
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

print(np.sum(arr, axis=0)) # This will add every rows and cols place from every sheets into a single combined sheet.
print(np.sum(arr, axis=1)) # This will add every rows from every single sheet.
print(np.sum(arr, axis=2)) # This will add every cols of each rows from every single sheet.

# max() , min()

print(np.max(arr, axis=0)) # This will give the largest values across all sheets and make it as a single largest sheet.
print(np.max(arr, axis=1)) # This will give the largest values across a sheet and make it as a single row.
print(np.max(arr, axis=2)) # This will give the largest values across all cols in a single row from every sheet.


print(np.min(arr, axis=0)) 
print(np.min(arr, axis=1)) 
print(np.min(arr, axis=2)) 
