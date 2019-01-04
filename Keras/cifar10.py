# -*- coding: utf-8 -*-
"""
Created on Fri Jan  4 14:47:27 2019

@author: DELL
"""

from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report
from keras.models import Sequential
from keras.layers.core import Dense
from keras.optimizers import SGD
from keras.datasets import cifar10

import matplotlib.pyplot as plt
import numpy as np

# load data
(X_train, y_train), (X_test, y_test) = cifar10.load_data()
X_train = X_train.astype('float')/255
X_test = X_test.astype('float')/255
X_train = X_train.reshape(X_train.shape[0], -1)
X_test = X_test.reshape(X_test.shape[0], -1)

# one-hot encoding
lb = LabelBinarizer()
y_train = lb.fit_transform(y_train)
y_test = lb.transform(y_test)

#label
labelNames = ['airplane','automobile','bird','cat','deer','dog','frog','house','ship','truck']

# model
model = Sequential()
model.add(Dense(1024, activation='relu', input_shape=[3072,]))
model.add(Dense(512, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(10, activation='softmax'))

# train
sgd = SGD(0.01)
model.compile(sgd, 'categorical_crossentropy', ['accuracy'])
H = model.fit(X_train, y_train, 64, 100, validation_data=(X_test, y_test))

# graph of loss and accuracy
fig = plt.figure()
plt.plot(np.arange(0, 100), H.history['loss'], label='training loss')
plt.plot(np.arange(0, 100), H.history['val_loss'], label='validation loss')
plt.plot(np.arange(0, 100), H.history['acc'], label='accuracy')
plt.plot(np.arange(0, 100), H.history['val_acc'], label='validation accuracy')
plt.title('Accuracy and Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss|Accuracy')
plt.legend()
