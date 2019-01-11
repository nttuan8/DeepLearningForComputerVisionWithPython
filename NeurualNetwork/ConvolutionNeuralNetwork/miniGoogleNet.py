# -*- coding: utf-8 -*-
"""
Created on Fri Jan 11 09:48:46 2019

@author: DELL
"""

from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.convolutional import AveragePooling2D
from keras.layers.core import Dense
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dropout

from keras.layers import Input
from keras.layers import concatenate

from keras.models import Model

class MiniGoogleNet:
    @staticmethod
    def conv_module(x, k, kx, ky, stride, axis, pad='same'):
        x = Conv2D(k, (kx, ky), strides=stride, padding=pad)(x)
        x = BatchNormalization(axis=axis)(x)
        x = Activation('relu')(x)
        return x
    
    @staticmethod
    def inception_module(x, numK1x1, numK3x3, axis):
        conv_1x1 = MiniGoogleNet.conv_module(x, numK1x1, 1, 1, (1,1), axis)
        conv_3x3 = MiniGoogleNet.conv_module(x, numK1x1, 3, 3, (1,1), axis)
        x = concatenate([conv_1x1, conv_3x3], axis=axis)        
        return x
    
    @staticmethod
    def dowsample_module(x, k, axis):
        conv_3x3 = MiniGoogleNet.conv_module(x, k, 3, 3, (2, 2), axis, pad='valid')
        pool = MaxPooling2D(pool_size=(3,3), strides=(2,2))(x)
        x = concatenate([conv_3x3, pool], axis=axis)
        return x
    
    @staticmethod
    def build(width, height, depth, classes):
        input_shape = (width, height, depth)
        axis = -1
        
        inputs = Input(shape=input_shape)
        x = MiniGoogleNet.conv_module(inputs, 93, 3, 3, (1,1), axis)
        x = MiniGoogleNet.inception_module(x, 32, 32, axis)
        x = MiniGoogleNet.inception_module(x, 32, 48, axis)
        x = MiniGoogleNet.dowsample_module(x, 80, axis)
        
        x = MiniGoogleNet.inception_module(x, 112, 48, axis)
        x = MiniGoogleNet.inception_module(x, 96, 64, axis)
        x = MiniGoogleNet.inception_module(x, 80, 80, axis)
        x = MiniGoogleNet.inception_module(x, 48, 96, axis)
        x = MiniGoogleNet.dowsample_module(x, 96, axis)
        
        x = MiniGoogleNet.inception_module(x, 176, 160, axis)
        x = MiniGoogleNet.inception_module(x, 176, 160, axis)
        x = AveragePooling2D((7,7))(x)
        x = Dropout(0.5)(x)
        
        x = Flatten()(x)
        x = Dense(classes)(x)
        x = Activation('softmax')(x)
        
        model = Model(inputs, x, name='googlenet')
        return model