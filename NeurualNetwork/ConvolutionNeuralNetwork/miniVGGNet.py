# -*- coding: utf-8 -*-
"""
Created on Sat Jan  5 10:09:46 2019

@author: DELL
"""
from keras import backend as K
from keras.models import Sequential
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Dense
from keras.layers import BatchNormalization
from keras.layers import Dropout
from keras.layers import Flatten

class MiniVGGNet:
    @staticmethod
    def build(width, height, depth, classes):
        input_shape = (height, width, depth)
        if K.image_data_format() == 'channels_first':
            input_shape = (depth, height, width)
        
        model = Sequential()
        
        model.add(Conv2D(32, (3,3), padding='same', input_shape=input_shape, activation='relu'))
        model.add(BatchNormalization())
        model.add(Conv2D(32, (3,3), padding='same', activation='relu'))
        model.add(BatchNormalization())
        model.add(MaxPooling2D((2,2), (2,2)))
        model.add(Dropout(0.25))
        
        model.add(Conv2D(64, (3,3), padding='same', activation='relu'))
        model.add(BatchNormalization())
        model.add(Conv2D(64, (3,3), padding='same', activation='relu'))
        model.add(BatchNormalization())
        model.add(MaxPooling2D((2,2), (2,2)))
        model.add(Dropout(0.25))
        
        model.add(Flatten())
        model.add(Dense(512, activation='relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.5))
        model.add(Dense(10, activation='softmax'))
        
        return model