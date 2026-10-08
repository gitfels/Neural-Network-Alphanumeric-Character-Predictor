import numpy as np
from importdata import load_data, split, forward_prop, get_predictions, get_accuracy
import time
from matplotlib import pyplot as plt

#setup the data, organize an array, and shuffle it.
data = load_data('data/training.csv')


#print(m,n)
#print statement for debugging the shape of the data


#The number of classes for the dataset.
NUM_CLASSES = 47


train_data, dev_data = split(data)

X_train = train_data[:, 1:].T / 255.0
Y_train = train_data[:, 0]
X_dev   = dev_data[:, 1:].T / 255.0
Y_dev   = dev_data[:, 0]

def initialize_parameters(hidden=128):
    W1 = np.random.rand(hidden,784) - 0.5
    B1 = np.zeros((hidden, 1))
    W2 = np.random.rand(NUM_CLASSES,hidden) - 0.5
    B2 = np.zeros((NUM_CLASSES, 1))

    return W1, B1, W2, B2




def onehot(Y):
    one_hot_Y = one_hot_Y = np.zeros((Y.size, NUM_CLASSES))
    one_hot_Y[np.arange(Y.size),Y] = 1

    return one_hot_Y.T
def back_prop(W1, B1, W2, B2,Z1, A1, Z2, A2, X,Y):
    onehotY = onehot(Y)
    dZ2 = A2 - onehotY
    dW2 = 1/X.shape[1] * dZ2.dot(A1.T)
    dB2 = 1/X.shape[1] * np.sum(dZ2, axis=1, keepdims=True)
    dZ1 = W2.T.dot(dZ2) * (Z1 > 0)     
    dW1 = 1/X.shape[1] * dZ1.dot(X.T)
    dB1 = 1/X.shape[1] * np.sum(dZ1, axis=1, keepdims=True)

    return dW1, dB1, dW2, dB2

def update_parameters(W1, B1, W2, B2, dW1, dB1, dW2,dB2, learning_rate):
    
    W1 = W1- learning_rate*dW1
    B1 = B1 - learning_rate*dB1
    W2 = W2 - learning_rate*dW2
    B2 = B2 - learning_rate*dB2
    return W1, B1, W2, B2



def grad_desc(X,Y,alpha,iterations, checkpoint=50, X_dev=None, Y_dev=None):
    W1, B1, W2, B2 = initialize_parameters()
    acc_history = []
    dev_acc_history = []
    iter_history=[]
    loss_history = []
    start_time = time.time()

    for i in range(iterations):
        Z1,A1,Z2,A2=forward_prop(W1,B1,W2,B2,X)
        dW1, dB1, dW2, dB2 = back_prop(W1, B1, W2, B2,Z1, A1, Z2, A2, X,Y)
        W1, B1, W2, B2 = update_parameters(W1, B1, W2, B2, dW1, dB1, dW2,dB2, alpha)

        
        

        if i%checkpoint == 0 or i == iterations-1:
            acc = get_accuracy(get_predictions(A2),Y)
            print(f"Accuracy after {i} iterations = ", acc)
            if X_dev is not None and Y_dev is not None:
                _, _, _, A2_dev = forward_prop(W1, B1, W2, B2, X_dev)
                dev_acc = get_accuracy(get_predictions(A2_dev), Y_dev)
                print(f"Dev accuracy after {i} iterations = ", dev_acc)
                dev_acc_history.append(dev_acc)
            
            print(f"Elapsed time: {time.time()-start_time:.2f} seconds")
            acc_history.append(acc)
            iter_history.append(i)
            losscalc = -np.mean(np.log(A2[Y, np.arange(Y.size)] + 1e-12))
            loss_history.append(losscalc)

    total_time = time.time() - start_time
    print(f"Training completed in {total_time:.2f} seconds.")
    #DISPLAY ACCURACY CURVE: comment out if you don't want to see this.
    plt.figure()
    plt.plot(iter_history, acc_history, marker = 'o', color = 'green', label = "Training Accuracy")
    if dev_acc_history:
        plt.plot(iter_history, dev_acc_history, marker = 'o', color = 'blue', label = "Dev Accuracy")
        plt.legend()
    plt.title("Accuracy vs. iterations")
    plt.xlabel("Iterations")
    plt.ylabel("Accuracy")
    plt.xlim(0, iterations)
    plt.ylim(0, 1)
    plt.savefig('graphs/training_accuracy.png', dpi=120)

    #DISPLAY LOSS CURVE: comment out if you don't want to see this.
    plt.figure()
    plt.plot(iter_history, loss_history, marker = 'o', color='red')
    plt.title("Training loss vs. iterations")
    plt.xlabel("Iterations")
    plt.ylabel("Training Loss")
    plt.xlim(0, iterations)
    plt.savefig('graphs/training_loss.png', dpi=120)


    return W1,B1,W2,B2

#gradient descent, storing data.
W1,B1,W2,B2 = grad_desc(X_train, Y_train, 0.2, 1000, 50, X_dev, Y_dev)
np.savez('emnist_params.npz', W1=W1, B1=B1, W2=W2, B2=B2)


#dev accuracy, to see how well the model performs on unseen data.
_, _, _, A2 = forward_prop(W1, B1, W2, B2, X_dev)
print("Dev accuracy:", get_accuracy(get_predictions(A2), Y_dev))



plt.show()