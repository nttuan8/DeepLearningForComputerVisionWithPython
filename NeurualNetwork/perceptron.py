# -*- coding: utf-8 -*-
"""
Created on Thu Jan  3 10:32:29 2019

@author: DELL
"""

import numpy as np

class Perceptron:
    def __init__(self, N, alpha=0.1):
        self.W = np.random.rand(N+1)/np.sqrt(N)
        self.alpha = alpha
        
    def step(self, x):
        if x > 0:
            return 1
        else:
            return 0
    def fit(self, X, y, epochs=10):
        X = np.c_[X, np.ones(X.shape[0])]
        for e in range(0, epochs):
            for (x, label) in zip(X,y):
                p = np.dot(x, self.W)
                predict = self.step(p)
                if(predict != label):
                    self.W += -self.alpha * x.T * (predict-label)
    
    def predict(self, X, addBias=True):
        X = np.atleast_2d(X)
        if addBias:
             X = np.c_[X, np.ones(X.shape[0])]
        return self.step(np.dot(X, self.W))

#perceptron or
X = np.array([[0,0], [0,1], [1,0], [1,1]])
y = np.array([0,1,1,1])
p = Perceptron(X.shape[1])
p.fit(X, y)
for (x, target) in zip(X, y):
    predict = p.predict(x)
    print('[INFO] Data = {}, ground-truth: {}, predict: {}'.format(x, target, predict))
    
#perceptron and
X = np.array([[0,0], [0,1], [1,0], [1,1]])
y = np.array([0,0,0,1])
p = Perceptron(X.shape[1])
p.fit(X, y)
for (x, target) in zip(X, y):
    predict = p.predict(x)
    print('[INFO] Data = {}, ground-truth: {}, predict: {}'.format(x, target, predict))
    
#perceptron xor
X = np.array([[0,0], [0,1], [1,0], [1,1]])
y = np.array([0,1,1,0])
p = Perceptron(X.shape[1])
p.fit(X, y)
for (x, target) in zip(X, y):
    predict = p.predict(x)
    print('[INFO] Data = {}, ground-truth: {}, predict: {}'.format(x, target, predict))