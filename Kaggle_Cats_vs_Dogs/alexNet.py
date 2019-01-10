# -*- coding: utf-8 -*-
"""
Created on Thu Jan 10 09:29:01 2019

@author: DELL
"""

from keras.models import Sequential
from keras.layers.core import Dense
from keras.layers.core import Dropout
from keras.layers.core import Flatten

from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D

from keras.layers.normalization import BatchNormalization

from keras.regularizers import l2

class AlexNet:
    @staticmethod
    def build(width, height, depth, classes, reg=0.0002):
        input_shape = (width, height, depth)
        model = Sequential()
        
        # BLOCK 1: CONV-RELU-POOL
        model.add(Conv2D(96, (11, 11), strides=(4, 4), padding='same', input_shape=input_shape, 
                         kernel_regularizer=l2(reg), activation='relu'))
        model.add(BatchNormalization())
        model.add(MaxPooling2D((3,3), (2,2)))
        model.add(Dropout(0.25))
        
        #BLOCK 2: second CONV-RELU-POOL
        model.add(Conv2D(256, (5,5), padding='same', activation='relu', kernel_regularizer=l2(reg)))
        model.add(BatchNormalization())
        model.add(MaxPooling2D((3,3), (2,2)))
        model.add(Dropout(0.25))
        
        #BLOCK 3: CONV-RELU-CONV-RELU-CONV-RELU
        model.add(Conv2D(383, (3,3), padding='same', activation='relu', kernel_regularizer=l2(reg)))
        model.add(BatchNormalization())
        
        model.add(Conv2D(383, (3,3), padding='same', activation='relu', kernel_regularizer=l2(reg)))
        model.add(BatchNormalization())
        
        model.add(Conv2D(256, (3,3), padding='same', activation='relu', kernel_regularizer=l2(reg)))
        model.add(BatchNormalization())
        
        model.add(MaxPooling2D((3,3), (2,2)))
        model.add(Dropout(0.25))
        
        # BLOCK 4: FC -> RELU
        model.add(Flatten())
        model.add(Dense(4096, kernel_regularizer=l2(reg), activation='relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.5))
        
        # BLOCK 5: second FC -> RELU
        model.add(Dense(4096, kernel_regularizer=l2(reg), activation='relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.5))
        
        model.add(Dense(classes, kernel_regularizer=l2(reg), activation='softmax'))
        
        return model
        
        