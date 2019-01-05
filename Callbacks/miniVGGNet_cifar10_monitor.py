# -*- coding: utf-8 -*-
"""
Created on Sat Jan  5 10:48:37 2019

@author: DELL
"""

from trainingmonitor import TrainingMonitor
import sys
sys.path.append("..")
from NeurualNetwork.ConvolutionNeuralNetwork.miniVGGNet import MiniVGGNet

from keras.datasets import cifar10
from keras.optimizers import SGD

from sklearn.preprocessing import LabelBinarizer

import matplotlib.pyplot as plt
import numpy as np
import os

(X_train, y_train), (X_test, y_test) = cifar10.load_data()
X_train = X_train.astype('float')/255
X_test = X_test.astype('float')/255

X_train = X_train[:100,:,:,:]
X_test = X_test[:100,:,:,:]
y_train = y_train[:100]
y_test = y_test[:100]


lb = LabelBinarizer()
y_train = lb.fit_transform(y_train)
y_test = lb.fit_transform(y_test)



numOfEpoch = 40

model = MiniVGGNet.build(32, 32, 3, 10)
model.compile(SGD(lr=0.01, momentum=0.9, nesterov=True), 'categorical_crossentropy', ['accuracy'])

folPath = 'Output'
jsonPath = os.path.sep.join([folPath, '{}.json'.format(os.getpid())])

callbacks = [TrainingMonitor(folPath, jsonPath)]

H = model.fit(X_train, y_train, 128, numOfEpoch, validation_data=(X_test, y_test), callbacks=callbacks)

# graph of loss and accuracy
fig = plt.figure()
plt.plot(np.arange(0, numOfEpoch), H.history['loss'], label='training loss')
plt.plot(np.arange(0, numOfEpoch), H.history['val_loss'], label='validation loss')
plt.plot(np.arange(0, numOfEpoch), H.history['acc'], label='accuracy')
plt.plot(np.arange(0, numOfEpoch), H.history['val_acc'], label='validation accuracy')
plt.title('Accuracy and Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss|Accuracy')
plt.legend()