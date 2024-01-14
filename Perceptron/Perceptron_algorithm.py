import numpy as np
import random

# activation functions
def unit_step_function(x):
    return np.where(x>=0, 1, 0)

def sgn_function(x):
    return np.where(x>=0, 1, -1)

# accuracy function
def accuracy(y_true, y_pred):
    accuracy = np.sum(y_true == y_pred) / len(y_true)
    return accuracy

class Perceptron:
    
    def __init__(self, learning_rate=0.01, iterations=1000, activation_function=unit_step_function, random_seed=42):
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.activation_function = activation_function
        self.weights = None
        self.bias = None
        self.random_seed = random_seed
        self.intercept_ = None
        self.coef_ = None
        

    def fit(self, X, y):
        samples, features = X.shape
        self.errors_ = []
        np.random.seed(self.random_seed)
        self.weights = np.random.uniform(-1, 1, features) # initialize weights with random values uniformly distributed between -1 and 1
        self.bias = random.uniform(-1.0, 1.0) # initialize bias with random values uniformly distributed between -1 and 1
        # set y_ according to the chosen activation function
        if self.activation_function == unit_step_function:
            y_ = np.where(y>=0, 1, 0)
        elif self.activation_function == sgn_function:
            y_ = np.where(y>=0, 1, -1)
        
        # learn weights
        for _ in range(self.iterations):
            errors = 0 # set current iteration errors to 0
            for idx, x_i in enumerate(X):
                linear_output = np.dot(x_i, self.weights) + self.bias # calculate the linear output features_values * weights + bias
                y_pred = self.activation_function(linear_output) # calculate the predicted output
                update = self.learning_rate * (y_[idx] - y_pred) # calculate the update values
                # update the weights and bias
                self.weights += update * x_i
                self.bias += update
                errors += int(update != 0.0) # count the number of errors
            self.errors_.append(errors) # append the number of errors to the errors_ list
        
        # set intercept and coeficients
        self.intercept_ = self.bias
        self.coef_ = self.weights
    
    def predict(self, X):
        linear_output = np.dot(X, self.weights) + self.bias
        y_pred = self.activation_function(linear_output)
        return y_pred