# -*- coding: utf-8 -*-
"""
Created on Sat Jan  5 09:07:15 2019

@author: DELL
"""

from keras.models import Sequential
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Dense
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras import backend as K

class LeNet:
    @staticmethod
    def build(width, height, depth, classes):
        input_shape = (height, width, depth)
        if K.image_data_format() == 'channels_first':
            input_shape = (depth, height, width)
        
        model = Sequential()
        
        model.add(Conv2D(20, (5,5), padding='same', input_shape=input_shape))
        model.add(Activation('tanh'))
        model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
        
        model.add(Conv2D(50, (5,5), padding='same'))
        model.add(Activation('tanh'))
        model.add(MaxPooling2D((2,2)))
        
        model.add(Flatten())
        model.add(Dense(500))
        model.add(Activation('tanh'))
        model.add(Dense(10))
        model.add(Activation('softmax'))
        
        return model
        