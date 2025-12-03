from tensorflow.keras.datasets import fashion_mnist
import numpy as np

(X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()

for i in range(10):
  print(f"No: {i}: with the train shape of {X_train[i].shape} has the value of {y_train[i]}")
  print(f"Min of X_train is {X_train.min()} and Max of X_train is {X_train.max()}")

print(f"Training data shape: {X_train.shape}")  
print(f"Test data shape: {X_test.shape}")      
print(f"Unique labels: {np.unique(y_train)}")   # 0-9 classes


X_train = X_train.astype('float32')/255.0
X_test  = X_test.astype('float32')/255.0

X_train = X_train.reshape(-1, 28, 28, 1)
X_test = X_train.reshape(-1, 28, 28, 1)

from tensorflow.keras.utils import to_categorical
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)