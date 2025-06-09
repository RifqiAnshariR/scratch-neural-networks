import numpy as np

EPOCHS = 100
BATCH_SIZE = 100
LEARNING_RATE = 0.001

data = np.load('./data/mnist_flattened.npz')
x_train = data['x_train']
y_train = data['y_train']

def relu(x):
    return np.maximum(0, x)

def relu_deriv(x):
    return (x > 0).astype(float)

def softmax(x):
    exp = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp / np.sum(exp, axis=1, keepdims=True)

def cross_entropy(y_true, y_pred):
    return -np.sum(y_true * np.log(y_pred + 1e-8)) / y_true.shape[0]

w1 = np.random.randn(784, 128) * 0.01
b1 = np.zeros((1, 128))
w2 = np.random.randn(128, 10) * 0.01
b2 = np.zeros((1, 10))

for epoch in range(EPOCHS):
    permutation = np.random.permutation(x_train.shape[0])
    x_shuffled = x_train[permutation]
    y_shuffled = y_train[permutation]
    
    for i in range(0, x_train.shape[0], BATCH_SIZE):
        x_batch = x_shuffled[i:i+BATCH_SIZE]
        y_batch = y_shuffled[i:i+BATCH_SIZE]
        
        # Forward
        # Layer 1
        z1 = np.dot(x_batch, w1) + b1
        a1 = relu(z1)
        # Layer 2
        z2 = np.dot(a1, w2) + b2
        a2 = softmax(z2)
        
        # Loss (Cross Entropy)
        loss = cross_entropy(y_batch, a2)
        
        # Backward
        # Layer 2
        dz2 = a2 - y_batch
        dw2 = np.dot(np.transpose(a1), dz2)
        db2 = np.sum(dz2, axis=0, keepdims=True)
        # Layer 1
        da1 = np.dot(dz2, np.transpose(w2))
        dz1 = da1 * relu_deriv(z1)
        dw1 = np.dot(np.transpose(x_batch), dz1)
        db1 = np.sum(dz1, axis=0, keepdims=True)
        
        # Rata-rata
        dw2 /= BATCH_SIZE
        db2 /= BATCH_SIZE
        dw1 /= BATCH_SIZE
        db1 /= BATCH_SIZE
        
        # Update
        w1 -= LEARNING_RATE * dw1
        b1 -= LEARNING_RATE * db1
        w2 -= LEARNING_RATE * dw2
        b2 -= LEARNING_RATE * db2

    print(f"Epoch: {epoch + 1}, Loss: {loss:.3f}")
    
np.savez('./data/model_trained.npz', w1=w1, b1=b1, w2=w2, b2=b2)
