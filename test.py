import numpy as np

def load_dataset(dataset_path):
    data = np.load(dataset_path)
    x_test = data['x_test']
    y_test = data['y_test']
    return x_test, y_test

def load_model(model_path):
    param = np.load(model_path)
    w1 = param['w1']
    b1 = param['b1']
    w2 = param['w2']
    b2 = param['b2']
    return w1, b1, w2, b2

def relu(x):
    return np.maximum(0, x)

def softmax(x):
    exp = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp / np.sum(exp, axis=1, keepdims=True)

def main():
    x_test, y_test = load_dataset('./data/mnist_flattened.npz')
    w1, b1, w2, b2 = load_model('./data/model_trained.npz')

    # Forward
    # Layer 1
    z1 = np.dot(x_test, w1) + b1
    a1 = relu(z1)
    # Layer 2
    z2 = np.dot(a1, w2) + b2
    a2 = softmax(z2)

    predictions = np.argmax(a2, axis=1)
    true_labels = np.argmax(y_test, axis=1)
    accuracy = np.mean(predictions == true_labels) * 100
    print(f"Accuracy: {accuracy:.2f}%")

if __name__ == "__main__":
    main()