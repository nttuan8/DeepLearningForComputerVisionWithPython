# -*- coding: utf-8 -*-
"""
Created on Sat Jan 12 09:50:50 2019

@author: DELL
"""

from sklearn.preprocessing import LabelBinarizer
from resnet import ResNet
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import LearningRateScheduler
from keras.optimizers import SGD
from keras.datasets import cifar10
import numpy as np
import matplotlib.pyplot as plt

NUM_EPOCHS = 80
INIT_LR = 1e-1

def poly_decay(epoch):
    maxEpoch = NUM_EPOCHS
    baseLR = INIT_LR
    power = 1.0
    alpha = baseLR * (1 - epoch / float(maxEpoch)) ** power
    # Return new learning rate    
    return alpha

(X_train, y_train), (X_test, y_test) = cifar10.load_data()

mean = np.mean(X_train, axis=0)
X_train -= mean.astype('uint8')
X_test -= mean.astype('uint8')

lb = LabelBinarizer()
y_train = lb.fit_transform(y_train)
y_test = lb.fit_transform(y_test)

aug = ImageDataGenerator(width_shift_range=0.1, height_shift_range=0.1, horizontal_flip=True, fill_mode='nearest')

opt = SGD(1e-1)
model = ResNet.build(32, 32, 3, 10, (9,9,9), (64, 64, 128, 256), reg=0.0005)
model.compile(opt, 'categorical_crossentropy', ['accuracy'])
H = model.fit_generator(aug.flow(X_train, y_train, batch_size=128), validation_data=(X_test, y_test),
                    steps_per_epoch=len(X_train)//128, epochs=10, verbose=1)