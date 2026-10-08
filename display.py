import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from importdata import forward_prop, get_predictions, get_accuracy, load_mapping, load_data, split

params = np.load('emnist_params.npz')
W1, B1, W2, B2 = params["W1"], params["B1"], params["W2"], params["B2"]

data = load_data('data/training.csv')
_, val_data = split(data)
chars = load_mapping('data/emnist-balanced-mapping.txt')

X_val = val_data[:, 1:].T / 255.0
Y_val = val_data[:, 0]

val_index=5
Z1val, A1val, Z2val, A2val = forward_prop(W1,B1,W2,B2,X_val[:, val_index,None])
print("Predicted label: ",chars[get_predictions(A2val)[0]])
print("Actual label: ", chars[int(Y_val[val_index])])

image_array = X_val[:,val_index].reshape(28,28)
plt.imshow(image_array, cmap='gray')
plt.show()

Z1val, A1val, Z2val, A2val = forward_prop(W1,B1,W2,B2,X_val)
val_acc = get_accuracy(get_predictions(A2val),Y_val)
print("Validation accuracy = ", val_acc)