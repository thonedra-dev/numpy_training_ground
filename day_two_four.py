import numpy as np

# --------------------------
# Original 3D array
# Shape: (2 sheets, 3 rows, 4 columns)
arr3 = np.arange(1, 25).reshape(2,3,4)
print("Original array (2,3,4):\n", arr3, "\n")

# --------------------------
# 1️⃣ Swap sheets (axis 0) and rows (axis 1)
swapped_0_1 = np.swapaxes(arr3, 0, 1)
print("Swap axes 0 & 1 (sheets <-> rows), shape:", swapped_0_1.shape)
print(swapped_0_1, "\n")

# --------------------------
# 2️⃣ Swap sheets (axis 0) and columns (axis 2)
swapped_0_2 = np.swapaxes(arr3, 0, 2)
print("Swap axes 0 & 2 (sheets <-> columns), shape:", swapped_0_2.shape)
print(swapped_0_2, "\n")

# --------------------------
# 3️⃣ Swap rows (axis 1) and columns (axis 2)
swapped_1_2 = np.swapaxes(arr3, 1, 2)
print("Swap axes 1 & 2 (rows <-> columns), shape:", swapped_1_2.shape)
print(swapped_1_2, "\n")

# --------------------------
# 4️⃣ Combine swaps: first swap 0 & 1, then swap 1 & 2
combined_swap = np.swapaxes(np.swapaxes(arr3, 0, 1), 1, 2)
print("Combined swap (0<->1 then 1<->2), shape:", combined_swap.shape)
print(combined_swap, "\n")


