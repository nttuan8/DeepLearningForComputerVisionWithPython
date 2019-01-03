# -*- coding: utf-8 -*-
"""
Created on Thu Jan  3 15:18:06 2019

@author: DELL
"""

import numpy as np

def sigmoid(x):
        return 1/(1+np.exp(-x))
 
    
def sigmoid_derivative(x):
        return x*(1-x)


class NeuralNetwork:
    def __init__(self, layers, alpha=0.1):
        self.layers = layers
        self.alpha = alpha
        self.W = []
        for i in range(0, len(layers)-2):
            w = np.random.randn(layers[i]+1, layers[i+1]+1)
            self.W.append(w/np.sqrt(layers[i]))
        w = np.random.randn(layers[len(layers)-2] + 1, layers[len(layers)-1])
        self.W.append(w/np.sqrt(layers[len(layers)-2]))
        
    def __repr__(self):
        return "Neural network [{}]".format("-".join(str(l) for l in self.layers))
    
    def fit_partial(self, x, y):
        A = [np.atleast_2d(x)]
        
        #foward
        out = A[-1]
        for i in range(0, len(self.layers) - 1):
            out = sigmoid(np.dot(out, self.W[i]))
            A.append(out)
        
        #backward
        error = A[-1] - y
        D = [error * sigmoid_derivative(A[-1])]
        for i in reversed(range(1, len(self.layers)-1)):
            delta = np.dot(D[-1], self.W[i].T)
            delta = delta * sigmoid_derivative(A[i])
            D.append(delta)
        
        #reverse D
        D = D[::-1]
        
        for i in range(0, len(self.layers)-1):
            self.W[i] -= self.alpha * np.dot(A[i].T, D[i])
        
    def fit(self, X, y, epochs=20, verbose=10):
        X = np.c_[X, np.ones(X.shape[0])]
        for epoch in range(0, epochs):
            for (x, target) in zip(X, y):
                self.fit_partial(x, target)
            if epoch % verbose == 0:
                loss = self.calculate_loss(x, target)
                print("Epoch {}, loss {}".format(epoch, loss))
    
    def predict(self, X, addBias=True):
        X = np.atleast_2d(X)
        if addBias:
            X = np.c_[X, np.ones(X.shape[0])]
        for i in range(0, len(self.layers) - 1):
            X = sigmoid(np.dot(X, self.W[i]))
        return X

    def calculate_loss(self, X, y):
        X = np.atleast_2d(X)
        predict = self.predict(X, addBias=False)
        return np.sum((predict-y)**2)/2
        
        
#perceptron xor -> solve non-linear problem
X = np.array([[0,0], [0,1], [1,0], [1,1]])
y = np.array([0,1,1,0])
p = NeuralNetwork([2, 2, 1], 1)
p.fit(X, y, 20000, 100)
for (x, target) in zip(X, y):
    predict = p.predict(x)
    print('[INFO] Data = {}, ground-truth: {}, predict: {}'.format(x, target, predict))