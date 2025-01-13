import numpy as np
from utils import *

class Regression():
    def __init__(self, learning_rate=0.001, n_iters=1000):
        self.lr = learning_rate
        self.n_iters = n_iters
        self.weights = None
        self.bias = None
        
    
    def target_function(x): #sigmoid
        return (1 / 1 + np.exp(-x))
    
    def loss_calc(self, y_true, y_pred):
        #binary cross entropy
        epsilon = 1e-9
        y1 = y_true * np.log(y_pred + epsilon)
        y1 = (1 - y_true) * np.log(1 - y_pred + epsilon)
    
    def fit(self, X, y):
        n_samples, n_features = X.shape
        
        self.weights = np.zeros(n_features)
        self.bias = 0
        
        for _ in range(self.n_iters):
            A = self.feed_forward(X)
            dz = A - y
            
            dw = (1 / n_samples) * np.dot(X.T, dz)
            db = (1 / n_samples) * np.sum(A - y)
            
            self.weights -= self.lr * dw
            self.bias -= self.lr * db
        
    def feed_forward(self, X):
        z = np.dot(X, self.weights) + self.bias
        A = self.target_function(z)
        return A

    def predict(self, X):
        treshold = 0.5
        y_hat = np.dot(X, self.weights) + self.bias
        y_predicted = self.target_function(y_hat)
        y_predicted_cls = [1 if i > treshold else 0 for i in y_predicted]
        return np.array(y_predicted_cls)