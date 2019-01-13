# -*- coding: utf-8 -*-
"""
Created on Sat Jan 12 22:00:38 2019

@author: DELL
"""

from keras.models import Sequential
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D

from keras.layers.core import Dense
from keras.layers.core import Dropout
from keras.layers.core import Activation
from keras.layers.core import Flatten

from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import ELU

class VGGModelEmotion:
    @staticmethod
    def build(width, height, depth, classes):
        input_shape = (height, width, depth)
        
        model = Sequential()
        
        model.add(Conv2D(32, (3,3), padding='same', kernel_initializer='he_normal', input_shape=input_shape))
        model.add(ELU())
        model.add(BatchNormalization())
        model.add(Conv2D(32, (3,3), padding='same', kernel_initializer='he_normal'))
        model.add(ELU())
        model.add(BatchNormalization())
        model.add(MaxPooling2D((2,2)))
        model.add(Dropout(0.25))
        
        model.add(Conv2D(64, (3,3), padding='same', kernel_initializer='he_normal'))
        model.add(ELU())
        model.add(BatchNormalization())
        model.add(Conv2D(64, (3,3), padding='same', kernel_initializer='he_normal'))
        model.add(ELU())
        model.add(BatchNormalization())
        model.add(MaxPooling2D((2,2)))
        model.add(Dropout(0.25))
        
        model.add(Conv2D(128, (3,3), padding='same', kernel_initializer='he_normal'))
        model.add(ELU())
        model.add(BatchNormalization())
        model.add(Conv2D(128, (3,3), padding='same', kernel_initializer='he_normal'))
        model.add(ELU())
        model.add(BatchNormalization())
        model.add(MaxPooling2D((2,2)))
        model.add(Dropout(0.25))
        
        model.add(Flatten())
        model.add(Dense(64, kernel_initializer='he_normal'))
        model.add(ELU())
        model.add(BatchNormalization())
        model.add(Dropout(0.5))
        
        model.add(Dense(64, kernel_initializer='he_normal'))
        model.add(ELU())
        model.add(BatchNormalization())
        model.add(Dropout(0.5))
        
        model.add(Dense(classes, kernel_initializer='he_normal'))
        model.add(Activation('softmax'))
        
        return model
        