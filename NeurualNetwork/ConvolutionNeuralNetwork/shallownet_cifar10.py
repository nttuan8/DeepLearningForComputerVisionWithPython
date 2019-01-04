# -*- coding: utf-8 -*-
"""
Created on Fri Jan  4 21:58:13 2019

@author: DELL
"""

from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report
from keras.optimizers import SGD
from keras.datasets import cifar10
from keras.models import load_model

import matplotlib.pyplot as plt
import numpy as np

from shallownet import ShallowNet

(X_train, y_train), (X_test, y_test) = cifar10.load_data()
X_train = X_train.astype('float')/255
X_test = X_test.astype('float')/255

lb = LabelBinarizer()
y_train = lb.fit_transform(y_train)
y_test = lb.transform(y_test)

model = ShallowNet.build(32, 32, 3, 10)
sgd = SGD()
model.compile(sgd, 'categorical_crossentropy', ['accuracy'])
H = model.fit(X_train, y_train, 64, 30, validation_data=(X_test, y_test))

#save and load model
model.save('./model/shallow_weight.hdf5')
model = load_model('./model/shallow_weight.hdf5')
# model.predict....

# graph of loss and accuracy
fig = plt.figure()
plt.plot(np.arange(0, 30), H.history['loss'], label='training loss')
plt.plot(np.arange(0, 30), H.history['val_loss'], label='validation loss')
plt.plot(np.arange(0, 30), H.history['acc'], label='accuracy')
plt.plot(np.arange(0, 30), H.history['val_acc'], label='validation accuracy')
plt.title('Accuracy and Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss|Accuracy')
plt.legend()