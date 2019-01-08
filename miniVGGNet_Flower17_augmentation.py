# -*- coding: utf-8 -*-
"""
Created on Tue Jan  8 16:28:59 2019

@author: DELL
"""

from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from Preprocessing.SimpleProcessor import SimplePreprocessor
from Preprocessing.AspectAwareProcessor import AspectAwareProcessor
from Preprocessing.ImageToArrayProcessor import ImageToArrayProcessor
from Dataset.SimpleDatasetLoader import SimpleDatasetLoader

from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator

from NeurualNetwork.ConvolutionNeuralNetwork.miniVGGNet import MiniVGGNet

from imutils import paths
import matplotlib.pyplot as plt
import numpy as np

imagePaths = list(paths.list_images("Dataset/oxfordflower17/jpg"))
print("[INFO]Load images")

aap = AspectAwareProcessor(64, 64)
imgToArr = ImageToArrayProcessor()
loader = SimpleDatasetLoader(processors=[aap, imgToArr])
data, label = loader.load(imagePaths, verbose=500)
data = data.astype('float')/255.

X_train, X_test, y_train, y_test = train_test_split(data, label, test_size=0.25, random_state=42)

lb = LabelBinarizer()
y_train = lb.fit_transform(y_train)
y_test = lb.fit_transform(y_test)

aug = ImageDataGenerator(rotation_range=30, width_shift_range=0.1, height_shift_range=0.1, shear_range=0.2, 
                         zoom_range=0.2, horizontal_flip=True, fill_mode='nearest')

numOfEpoch = 100
model = MiniVGGNet.build(64, 64, 3, 17)
model.compile(SGD(0.01, 0.9, nesterov=True), loss = 'categorical_crossentropy', metrics=['accuracy'])
H = model.fit_generator(aug.flow(X_train, y_train, batch_size=32), validation_data=(X_test, y_test), steps_per_epoch=32, epochs=numOfEpoch)

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

