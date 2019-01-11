# -*- coding: utf-8 -*-
"""
Created on Fri Jan 11 10:12:49 2019

@author: DELL
"""

from sklearn.preprocessing import LabelBinarizer
from miniGoogleNet import MiniGoogleNet
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import LearningRateScheduler
from keras.optimizers import SGD
from keras.datasets import cifar10

NUM_EPOCHS = 70
INIT_LR = 5e-3

def poly_decay(epoch):
    maxEpoch = NUM_EPOCHS
    baseLR = INIT_LR
    power = 1.0
    alpha = baseLR * (1 - epoch / float(maxEpoch)) ** power
    # Return new learning rate    
    return alpha

(X_train, y_train), (X_test, y_test) = cifar10.load_data()
X_train = X_train.astype('float')/255
X_test = X_test.astype('float')/255

lb = LabelBinarizer()
y_train = lb.fit_transform(y_train)
y_test = lb.fit_transform(y_test)

aug = ImageDataGenerator(width_shift_range=0.1, height_shift_range=0.1, horizontal_flip=True, fill_mode='nearest')

callbacks = [LearningRateScheduler(poly_decay)]

opt = SGD(lr = INIT_LR, momentum=0.9)
model = MiniGoogleNet.build(32, 32, 3, 10)
model.compile(opt, 'categorical_crossentropy', ['accuracy'])
H = model.fit_generator(aug.flow(X_train, y_train, batch_size=64), validation_data=(X_test, y_test),
                    steps_per_epoch=len(X_train)//64, epochs=NUM_EPOCHS, callbacks=callbacks, verbose=1)