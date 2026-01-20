# XOR
import numpy as np

def relu(x):
    return np.maximum(0, x)

def relu_deriv(x):
    return np.where(x>0, 1, 0)

X = np.array([[0,0],[0,1],[1,0],[1,1]]) #(4,2)
y = np.array([[0],[1],[1],[0]]) #(4,1)

np.random.seed(42)

input_layer_size = X.shape[1] #2
hidden_layer_size = 2 
output_layer_size = y.shape[1] #1

W1 = np.random.randn(input_layer_size, hidden_layer_size) #(2,2)
b1 = np.zeros((1, hidden_layer_size)) #(1,2)
W2 = np.random.randn(hidden_layer_size, output_layer_size) #(2,1)
b2 = np.zeros((1, output_layer_size)) #(1,1)

learning_r = 0.1
epochs = 10000

for epoch in range(epochs):
    # forward
    hidden_layer = np.dot(X, W1) + b1 #(4,2)dot(2,2)+(1,2) > (4,2)
    hidden_layer_activation = relu(hidden_layer) #(4,2)

    output_layer = np.dot(hidden_layer_activation, W2) + b2 #(4,2)dot(2,1)+(1,1) > (4,1)

    # computing error
    error = output_layer - y #(4,1)

    # backward
    d_output_layer = error 
    d_W2 = hidden_layer_activation.T.dot(d_output_layer) # dim=W2(2,1)
    d_b2 = np.mean(d_output_layer, axis=0, keepdims=True) # mean(1,1)

    d_hidden_layer_activated = d_output_layer.dot(W2.T) #(4,2)
    d_hidden_layer = d_hidden_layer_activated*relu_deriv(hidden_layer) # elementwise relu(4,2)
    d_W1 = X.T.dot(d_hidden_layer) #(2,2)
    d_b1 = np.mean(d_hidden_layer, axis=0, keepdims=True) # mean(1,2)

    # update
    W1 -= d_W1*learning_r; b1 -= d_b1*learning_r
    W2 -= d_W2*learning_r; b2 -= d_b2*learning_r

Xt = np.array([[0,0],[0,1],[1,1],[1,0]])

print(relu(Xt.dot(W1)+b1).dot(W2)+b2)





    



