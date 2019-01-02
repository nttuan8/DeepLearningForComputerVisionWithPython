# -*- coding: utf-8 -*-
"""
Created on Wed Jan  2 17:10:59 2019

@author: DELL
"""

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.datasets import make_blobs
import numpy as np
import matplotlib.pyplot as plt
import argparse


def sigmoid_activation(x):
    return 1.0/(1+np.exp(-x))

def predict(X, W):
    pred = sigmoid_activation(np.dot(X, W))
    pred[pred>0.5] = 1
    pred[pred<=0.5] = 0
    return pred


#ap = argparse.ArgumentParser()
#ap.add_argument("-e", "--epoch", default=100, type=float, help="# of epoches")
#ap.add_argument("-lr", "--learning_rate", default=0.01, type=float, help="learning rate")
#args = vars(ap.parse_args())
#
##get arguments
#epoches = args['epoch']
#learning_rate = args['learning_rate']

epoches = 100
learning_rate = 0.001

(X, y) = make_blobs(n_samples=1000, n_features=2, centers=2, cluster_std=1.5, random_state=1)
y = y.reshape(y.shape[0],1)

X = np.c_[X, np.ones((X.shape[0]))]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=43)

print("[INFO] Training")
W = np.random.randn(X.shape[1], 1)
losses = []

for e in range(0, epoches):
    result = sigmoid_activation(np.dot(X_train, W))
    error = result - y_train
    loss = np.sum(error**2)
    losses.append(loss)
    gradient = 2*X_train.T.dot(result*(1-result)*error)
    W += -learning_rate * gradient
    if e % 5 == 0:
        print("[INFO] Epoch {}, loss {:.7f}".format(int(e+1), loss))

print("[INFO] evalute")
y_predict = predict(X_test, W)
print(classification_report(y_test, y_predict))

#plot data
plt.figure()
plt.scatter(X_test[:, 0], X_test[:, 1], marker='o')
plt.show()

#plot loss function
plt.figure()
plt.plot(range(0, epoches), losses)
plt.title("Traing loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.show()