import numpy as np
import pandas as pd

def load_data(path, seed=1233):
    data = pd.read_csv(path, header=None, dtype=np.uint8).to_numpy()
    imgs = data[:, 1:].reshape(-1, 28, 28).transpose(0, 2, 1).reshape(-1, 784)
    data = np.column_stack([data[:, 0], imgs])
    np.random.seed(seed)
    np.random.shuffle(data)
    return data

def split(data, frac=0.8):
    k = int(frac * len(data))
    return data[:k], data[k:]

def softmax(Z):
    e = np.exp(Z - np.max(Z, axis=0, keepdims=True))
    return e / np.sum(e, axis=0, keepdims=True)

def forward_prop(W1, B1, W2, B2, X):
    Z1 = W1.dot(X) + B1
    A1 = np.maximum(0, Z1)
    Z2 = W2.dot(A1) + B2
    A2 = softmax(Z2)
    return Z1, A1, Z2, A2

def get_predictions(A2):
    return np.argmax(A2, 0)

def get_accuracy(predictions, Y):
    return np.sum(predictions == Y) / Y.size

def load_mapping(path):
    mapping = np.loadtxt(path, dtype=int)
    return {int(l): chr(int(c)) for l, c in mapping}