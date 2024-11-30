# firts try to implement a neural network

import numpy as np

np.random.seed(0)

# create a toy dataset

X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [0]])

def initialize_parameters(input_size, hidden_size, output_size):
    np.random.seed(0)
    W1 = np.random.randn(hidden_size, input_size) * 0.01
    b1 = np.zeros((hidden_size, 1))
    W2 = np.random.randn(output_size, hidden_size) * 0.01
    b2 = np.zeros((output_size, 1))
    parameters = {"W1": W1, "b1": b1, "W2": W2, "b2": b2}
    return parameters

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def forward_propagation(X, parameters):
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]

    Z1 = np.dot(W1, X.T) + b1
    A1 = sigmoid(Z1)
    Z2 = np.dot(W2, A1) + b2
    A2 = sigmoid(Z2)

    cache = {"Z1": Z1, "A1": A1, "Z2": Z2, "A2": A2}
    return A2, cache

def binary_cross_entropy_loss(A2, y):
    m = y.shape[0]
    epsilon = 1e-15  # Small constant to avoid log(0)
    A2 = np.clip(A2, epsilon, 1 - epsilon)  # Clip values to avoid log(0)
    loss = -(1/m) * np.sum(y*np.log(A2) + (1-y)*np.log(1-A2))
    return loss

def backward_propagation(parameters, cache, X, y):
    m = X.shape[0]
    W2 = parameters["W2"]
    A1 = cache["A1"]
    A2 = cache["A2"]

    dZ2 = A2 - y.T
    dW2 = (1/m) * np.dot(dZ2, A1.T)
    db2 = (1/m) * np.sum(dZ2, axis=1, keepdims=True)
    dZ1 = np.dot(W2.T, dZ2) * (A1 * (1 - A1))
    dW1 = (1/m) * np.dot(dZ1, X)
    db1 = (1/m) * np.sum(dZ1, axis=1, keepdims=True)

    gradients = {"dW1": dW1, "db1": db1, "dW2": dW2, "db2": db2}
    return gradients

def update_parameters(parameters, gradients, learning_rate):
    for key in parameters:
        parameters[key] -= learning_rate * gradients["d" + key]
    return parameters

def train(X, y, hidden_layer_size, num_iterations, learning_rate):
    parameters = initialize_parameters(X.shape[1], hidden_layer_size, 1)
    for i in range(num_iterations):
        A2, cache = forward_propagation(X, parameters)
        loss = binary_cross_entropy_loss(A2, y)
        gradients = backward_propagation(parameters, cache, X, y)
        parameters = update_parameters(parameters, gradients, learning_rate)
        if i % 1000 == 0:
            print(f"Iteration {i}: loss = {loss}")
    return parameters

parameters = train(X, y, hidden_layer_size=4, num_iterations=10000, learning_rate=0.1)

def predict(X, parameters):
    A2, _ = forward_propagation(X, parameters)
    predictions = (A2 > 0.5).astype(int)
    return predictions.T

predictions = predict(X, parameters)
print("Predictions:", predictions)
