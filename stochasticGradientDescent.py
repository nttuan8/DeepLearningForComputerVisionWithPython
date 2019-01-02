# -*- coding: utf-8 -*-
"""
Created on Wed Jan  2 21:52:37 2019

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


def getNext(X, y, batch_size):
    for i in range(0, X.shape[0], batch_size):
        yield(X[i:i+batch_size], y[i:i+batch_size])

#ap = argparse.ArgumentParser()
#ap.add_argument("-e", "--epoch", default=100, type=float, help="# of epoches")
#ap.add_argument("-lr", "--learning_rate", default=0.01, type=float, help="learning rate")
#ap.add_argument("-b", "--batch_size", default=50, type=float, help="batch size")
#args = vars(ap.parse_args())
#
##get arguments
#epoches = args['epoch']
#learning_rate = args['learning_rate']

epoches = 100
batch_size = 50
learning_rate = 0.01

(X, y) = make_blobs(n_samples=1000, n_features=2, centers=2, cluster_std=1.5, random_state=1)
y = y.reshape(y.shape[0],1)

X = np.c_[X, np.ones((X.shape[0]))]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=43)

print("[INFO] Training")
W = np.random.randn(X.shape[1], 1)
losses = []

for e in range(0, epoches):
    for (X_batch, y_batch) in getNext(X_train, y_train, batch_size):
        result = sigmoid_activation(np.dot(X_batch, W))
        error = result - y_batch
        loss = np.sum(error**2)
        losses.append(loss)
        gradient = 2*X_batch.T.dot(result*(1-result)*error)
        W += -learning_rate * gradient
        if e % 5 == 0:
            print("[INFO] Epoch {}, loss {:.7f}".format(int(e+1), loss))

print("[INFO] evalute")
y_predict = predict(X_test, W)
print(classification_report(y_test, y_predict))

#plot data
#plt.figure()
#plt.scatter(X_test[:, 0], X_test[:, 1], marker='o')
#plt.show()

#plot loss function
plt.figure()
plt.plot(range(0, len(losses)), losses)
plt.title("Traing loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.show()