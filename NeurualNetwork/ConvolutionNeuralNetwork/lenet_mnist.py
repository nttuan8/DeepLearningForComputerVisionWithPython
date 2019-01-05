# -*- coding: utf-8 -*-
"""
Created on Sat Jan  5 09:21:14 2019

@author: DELL
"""

from keras.optimizers import SGD
from keras import backend as K
from keras.models import load_model

from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn import datasets

import matplotlib.pyplot as plt
import numpy as np

from lenet import LeNet

dataset = datasets.fetch_mldata('MNIST Original')
data = dataset.data

if K.image_data_format() == 'channels_first':
    data = data.reshape(data.shape[0], 1, 28, 28)
else:
    data = data.reshape(data.shape[0], 28, 28, 1)

(X_train, X_test, y_train, y_test) = train_test_split(data, dataset.target.astype('int'), test_size=0.25)
X_train = X_train.astype('float')/255
X_test = X_test.astype('float')/255

lb = LabelBinarizer()
y_train = lb.fit_transform(y_train)
y_test = lb.transform(y_test)

model = LeNet.build(28, 28, 1, 10)
sgd = SGD()
model.compile(sgd, 'categorical_crossentropy', ['accuracy'])
H = model.fit(X_train, y_train, 64, 20, validation_data=(X_test, y_test))

#save and load model
#model.save('./model/lenet_mnist_weight.hdf5')
#model = load_model('./model/lenet_mnist_weight.hdf5')
# model.predict....

# graph of loss and accuracy
fig = plt.figure()
plt.plot(np.arange(0, 20), H.history['loss'], label='training loss')
plt.plot(np.arange(0, 20), H.history['val_loss'], label='validation loss')
plt.plot(np.arange(0, 20), H.history['acc'], label='accuracy')
plt.plot(np.arange(0, 20), H.history['val_acc'], label='validation accuracy')
plt.title('Accuracy and Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss|Accuracy')
plt.legend()