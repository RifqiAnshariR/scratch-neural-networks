# type: ignore
import os

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.makedirs('data', exist_ok=True)

import numpy as np
from tensorflow.keras.datasets import mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape(-1, 784) / 255.0
x_test = x_test.reshape(-1, 784) / 255.0
# One-Hot Encoding
y_train = np.eye(10)[y_train]
y_test = np.eye(10)[y_test]

np.savez_compressed('data/mnist_flattened.npz',
                    x_train=x_train, y_train=y_train,
                    x_test=x_test, y_test=y_test)

print("MNIST loaded!")
print("x_train shape:", x_train.shape)
print("y_train shape:", y_train.shape)
