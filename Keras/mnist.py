# -*- coding: utf-8 -*-
"""
Created on Fri Jan  4 10:37:36 2019

@author: DELL
"""

from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn import datasets

from keras.models import Sequential
from keras.layers.core import Dense
from keras.optimizers import SGD

import matplotlib.pyplot as plt
import numpy as np

# load dataset
dataset = datasets.fetch_mldata('MNIST Original')

# scale pixel to range [0,1]
data = dataset.data.astype('float')/255

# split train, test
(X_train, X_test, y_train, y_test) = train_test_split(data, dataset.target, test_size=0.25)

# one-hot encoding label
lb = LabelBinarizer()
y_train = lb.fit_transform(y_train)
y_test = lb.transform(y_test)

# model
model = Sequential()
model.add(Dense(256, input_shape=(784, ), activation='sigmoid'))
model.add(Dense(128, activation='sigmoid'))
model.add(Dense(10, activation='softmax'))

#train
sgd = SGD(0.01)
model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])
H = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=100, batch_size=128)

plt.figure()
plt.plot(np.arange(0, 100), H.history['loss'], label='train loss')
plt.plot(np.arange(0, 100), H.history['val_loss'], label='validation loss')
plt.plot(np.arange(0, 100), H.history['acc'], label='accuracy')
plt.plot(np.arange(0, 100), H.history['val_acc'], label='validation accuracy')
plt.title('Loss and accuracy')
plt.xlabel('Epoch')
plt.ylabel('Loss/accuracy')
plt.legend()
