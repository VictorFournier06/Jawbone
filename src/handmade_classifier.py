#imports
import numpy as np

#hyperparameters
input_dim = X_train[0].shape[0]
output_dim = 1
hidden_dim = 128
learning_rate = 0.001
epochs = 1000

#randomly initialize the weights and initalize biases at 0
W1 = np.random.normal(0, 0.1, (hidden_dim, input_dim))
b1 = np.zeros((hidden_dim, 1))
W2 = np.random.normal(0, 0.1, (output_dim, hidden_dim))
b2 = np.zeros((output_dim, 1))

#activation functions
epsilon = 1e-8

def relu(x):
    return(np.maximum(0, x))

def sigmoid(x):
    x = np.maximum(epsilon, x)
    return(1/(1+np.exp(-x)))

#training (10 seconds per epoch)
for epoch in range(epochs):
    #forward pass
    z1 = W1 @ X_train.T + b1
    a1 = relu(z1)
    z2 = W2 @ a1 + b2
    a2 = sigmoid(z2)
    #compute the binary cross entropy loss
    a2 = np.maximum(epsilon, a2)
    a2 = np.minimum(1-epsilon, a2)
    loss = -np.mean(y_train * np.log(a2) + (1-y_train) * np.log(1-a2))
    #backward pass
    dz2 = a2 - y_train
    dW2 = dz2 @ a1.T
    db2 = np.sum(dz2, axis=1, keepdims=True)
    da1 = W2.T @ dz2
    dz1 = da1 * (z1 > 0)
    dW1 = dz1 @ X_train
    db1 = np.sum(dz1, axis=1, keepdims=True)
    #update the weights
    W1 -= learning_rate * dW1
    b1 -= learning_rate * db1
    W2 -= learning_rate * dW2
    b2 -= learning_rate * db2
    #print the loss
    if epoch % (epochs//10) == 0:
        print("epoch", epoch, "loss", loss)
        
print(a2, y_train)
print(a2.shape, y_train.shape)

#testing
z1 = W1 @ np.array(X_test).T + b1
a1 = relu(z1)
z2 = W2 @ a1 + b2
a2 = sigmoid(z2)
#compute the loss
loss = -np.mean(y_test*np.log(a2) + (1-y_test)*np.log(1-a2))
print("loss", loss)