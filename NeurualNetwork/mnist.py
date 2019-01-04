# -*- coding: utf-8 -*-
"""
Created on Fri Jan  4 09:45:02 2019

@author: DELL
"""
import numpy as np
from neuralNetwork import NeuralNetwork
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn import datasets

print('[INFO] Load MNIST dataset')
digits = datasets.load_digits()
data = digits.data.astype('float')
data = (data - data.min())/(data.max()-data.min())
print('[INFO] Sample {}, dim {}'.format(data.shape[0], data.shape[1]))

#split train, test set
X_train, X_test, y_train, y_test = train_test_split(data, digits.target, test_size=0.25)

#convert label one-hot encoding
y_train = LabelBinarizer().fit_transform(y_train)
y_test = LabelBinarizer().fit_transform(y_test)

print('[INFO] Train neural network')
nn = NeuralNetwork([X_train.shape[1], 32, 16, 10])
print('[INFO] Neural network info {}'.format(nn))
nn.fit(X_train, y_train, epochs=1000)

print('[INFO] Evaluate')
predict = nn.predict(X_test)
predict = np.argmax(predict, axis=1)
print(classification_report(np.argmax(y_test, axis=1), predict))