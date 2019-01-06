# -*- coding: utf-8 -*-
"""
Created on Sun Jan  6 16:42:53 2019

@author: DELL
"""

from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from keras.preprocessing.image import img_to_array
from keras.optimizers import SGD
from lenet import LeNet
from utils.captchaHelper import preprocessing

from imutils import paths
import matplotlib.pyplot as plt
import numpy as np

import cv2
import os

data = []
labels = []

for imagePath in paths.list_images('Dataset'):
        image = cv2.imread(imagePath)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        image = preprocessing(image, 28, 28)
        image = img_to_array(image)
        data.append(image)
        label = imagePath.split(os.path.sep)[-2]
        labels.append(label)
        
data = np.array(data, dtype='float')/255.0
labels = np.array(labels)

(X_train, X_test, y_train, y_test) = train_test_split(data, labels, test_size=0.25, random_state=42)

lb = LabelBinarizer()
y_train = lb.fit_transform(y_train)
y_test = lb.fit_transform(y_test)

numOfEpoch = 5

model = LeNet.build(28, 28, 1, 9)
model.compile(SGD(0.01, momentum=0.9, nesterov=True), loss='categorical_crossentropy', metrics=['accuracy'])
H = model.fit(X_train, y_train, 32, numOfEpoch, validation_data=(X_test, y_test))

prediction = model.predict(X_test, batch_size=32)
print(classification_report(y_test.argmax(1), prediction.argmax(1), target_names = lb.classes_))

model.save('captcha_lenet.hdf5')

fig = plt.figure()
plt.plot(np.arange(0, numOfEpoch), H.history['loss'], label='training loss')
plt.plot(np.arange(0, numOfEpoch), H.history['val_loss'], label='validation loss')
plt.plot(np.arange(0, numOfEpoch), H.history['acc'], label='accuracy')
plt.plot(np.arange(0, numOfEpoch), H.history['val_acc'], label='validation accuracy')
plt.title('Accuracy and Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss|Accuracy')
plt.legend()
