# -*- coding: utf-8 -*-
"""
Created on Sun Jan  6 21:57:55 2019

@author: DELL
"""

from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from keras.optimizers import SGD
from keras.models import load_model
from miniVGGNet import MiniVGGNet

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import cv2


train = pd.read_csv('dataset.csv')
dataset = train.iloc[:, 1].values

data = np.zeros([dataset.shape[0], 48, 48, 1])
for i in range(0, dataset.shape[0]):
    strData = dataset[i]
    result = strData.split(' ')
    result = np.array(result).reshape(48, 48, 1)
    data[i] = result

labels = train.iloc[:, 0].values

data = np.array(data, dtype='float')/255.0
labels = np.array(labels)

(X_train, X_test, y_train, y_test) = train_test_split(data, labels, test_size=0.25, random_state=42)

lb = LabelBinarizer()
y_train = lb.fit_transform(y_train)
y_test = lb.fit_transform(y_test)

numOfEpoch = 40

model = MiniVGGNet.build(48, 48, 1, 7)
model.compile(SGD(0.1, momentum=0.9, nesterov=True), loss='categorical_crossentropy', metrics=['accuracy'])
H = model.fit(X_train, y_train, 64, numOfEpoch, validation_data=(X_test, y_test))

model = load_model('miniVGG_emotion.hdf5')
prediction = model.predict(X_test, batch_size=32)
print(classification_report(y_test.argmax(1), prediction.argmax(1)))


fig = plt.figure()
plt.plot(np.arange(0, numOfEpoch), H.history['loss'], label='training loss')
plt.plot(np.arange(0, numOfEpoch), H.history['val_loss'], label='validation loss')
plt.plot(np.arange(0, numOfEpoch), H.history['acc'], label='accuracy')
plt.plot(np.arange(0, numOfEpoch), H.history['val_acc'], label='validation accuracy')
plt.title('Accuracy and Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss|Accuracy')
plt.legend()
