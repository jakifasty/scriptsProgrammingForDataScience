#definition of different functions for forward propagation

import numpy as np

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def d_sigmoid(output):
    return output * (1 - output)

def forward_propagation(a_0, w_0, w_1):
    z_1 = np.matmul(a_0, w_0) #multiplication between the input and the weights of the 
    a_1 = sigmoid(z_1) #multiplication between the output of the first layer and the weights of the second layer

    z_2 = np.matmul(a_1, w_1)
    a_2 = sigmoid(z_2)

    return a_1, a_2 #return the output of the first layer, hidden layer value, and the output of the second layer, predicted value

def back_propagation(w_1, a_1, a_2, y): 
    a_2_error = a_2 - y #see how much mistakes we do in prediction
    #these are the mistakes we've done in the second layer
    layer_2_delta = np.multiply(a_2_error, d_sigmoid(a_2)) #d_sigmoid is the derivative of the sigmoid function
    
    layer_1_error = np.matmul(layer_2_delta, w_1.T)
    #these are the mistakes we've done in the first layer
    layer_1_delta = np.multiply(layer_1_error, d_sigmoid(a_1)) #d_sigmoid is the derivative of the sigmoid function
    
    return layer_1_delta, layer_2_delta

def train(x, y, hidden_size, alpha=1, num_iter = 60000):
    w_0 = 2 * np.random.random((x.shape[0], hidden_size)) - 1
    w_1 = 2 * np.random.random((hidden_size, 1)) - 1

    for i in range(num_iter):
        a_0 = x
        a_1, a_2 = forward_propagation(a_0, w_0, w_1)

        layer_1_delta, layer_2_delta = back_propagation(w_1, a_1, a_2, y)
    
        w_1 -= alpha * np.matmul(a_1.T, layer_2_delta)
        w_0 -= alpha * np.matmul(a_0.T, layer_1_delta)
    
    return w_0, w_1

def forward_propagation(a_0, w_0, w_1, dropout_rate=1):
    z_1 = np.matmul(a_0, w_0) #multiplication between the input and the weights of the
    a_1 = sigmoid(z_1) #multiplication between the output of the first layer and the weights of the second layer

    if dropout_rate != 0: #if dropout rate is not 0, we apply dropout
        a_1 *= np.random.binomial([np.ones((len(a_1), len(a_1[0])))], 1 - dropout_rate)[0] * (1.0 / (1 - dropout_rate))

    z_2 = np.matmul(a_1, w_1)
    a_2 = sigmoid(z_2)

    return a_1, a_2 #return the output of the first layer, hidden layer value, and the output of the second layer, predicted value
