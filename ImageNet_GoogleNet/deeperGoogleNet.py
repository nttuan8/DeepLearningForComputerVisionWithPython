# -*- coding: utf-8 -*-
"""
Created on Fri Jan 11 15:10:53 2019

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

from keras.regularizers import l2

from keras.models import Model

class DeeperGoogleNet:
    @staticmethod
    def conv_module(x, k, kx, ky, stride, axis, pad='same', reg=0.0005, name=None):
        if name != None:
            convName = name+'_conv'
            bnName = name+'_bn'
            actName = name+'_act'
        x = Conv2D(k, (kx, ky), strides=stride, padding=pad, kernel_regularizer=l2(reg), name=convName)(x)
        x = BatchNormalization(axis=axis, name=bnName)(x)
        x = Activation('relu', name=actName)(x)
        return x
    
    @staticmethod
    def inception_module(x, num1x1, num3x3Reduce, num3x3, num5x5Reduce, num5x5,
                         num1x1Proj, axis, stage, reg=0.005):
        first = DeeperGoogleNet.conv_module(x, num1x1, 1, 1, (1,1), axis, reg=reg, name=stage+'_first')
        
        second = DeeperGoogleNet.conv_module(x, num3x3Reduce, 1, 1, (1,1), axis, reg=reg, name=stage+'_second1')
        second = DeeperGoogleNet.conv_module(second, num3x3, 3, 3, (1,1), axis, reg=reg, name=stage+'_second2')
        
        third = DeeperGoogleNet.conv_module(x, num5x5Reduce, 1, 1, (1,1), axis, reg=reg, name=stage+'_third1')
        third = DeeperGoogleNet.conv_module(third, num5x5, 5, 5, (1,1), axis, reg=reg, name=stage+'_third2')
        
        fourth = MaxPooling2D((3,3), strides=(1,1), padding='same', name=stage+'_pool')(x)
        fourth = DeeperGoogleNet.conv_module(fourth, num1x1Proj, 1, 1, (1,1), axis, reg=reg, name=stage+'_fourth')
        
        x = concatenate([first, second, third, fourth], axis=axis, name=stage+'_fixed')        
        return x
    
    @staticmethod
    def dowsample_module(x, k, axis):
        conv_3x3 = DeeperGoogleNet.conv_module(x, k, 3, 3, (2, 2), axis, pad='valid')
        pool = MaxPooling2D(pool_size=(3,3), strides=(2,2))(x)
        x = concatenate([conv_3x3, pool], axis=axis)
        return x
    
    @staticmethod
    def build(width, height, depth, classes, reg=0.005):
        input_shape = (width, height, depth)
        axis = -1
        
        inputs = Input(shape=input_shape)
        x = DeeperGoogleNet.conv_module(inputs, 64, 5, 5, (1,1), axis, reg=reg, name='block1')
        x = MaxPooling2D((3,3), strides=(2,2), padding='same', name='pool1')(x)
        x = DeeperGoogleNet.conv_module(x, 64, 1, 1, (1, 1), axis, reg=reg, name="block2")
        x = DeeperGoogleNet.conv_module(x, 192, 3, 3, (1, 1), axis, reg=reg, name="block3")
        x = MaxPooling2D((3,3), strides=(2,2), padding='same', name='pool2')(x)
        x = DeeperGoogleNet.inception_module(x, 64, 96, 128, 16, 32, 32, axis, "3a", reg=reg)
        x = DeeperGoogleNet.inception_module(x, 128, 128, 192, 32, 96, 64, axis, "3b", reg=reg)
        x = MaxPooling2D((3, 3), strides=(2, 2), padding="same", name="pool3")(x)
        x = DeeperGoogleNet.inception_module(x, 192, 96, 208, 16, 48, 64, axis, "4a", reg=reg)
        x = DeeperGoogleNet.inception_module(x, 160, 112, 224, 24, 64, 64, axis, "4b", reg=reg)
        x = DeeperGoogleNet.inception_module(x, 128, 128, 256, 24, 64, 64, axis, "4c", reg=reg)
        x = DeeperGoogleNet.inception_module(x, 112, 144, 288, 32, 64, 64, axis, "4d", reg=reg)
        x = DeeperGoogleNet.inception_module(x, 256, 160, 320, 32, 128, 128, axis, "4e", reg=reg)
        x = MaxPooling2D((3, 3), strides=(2, 2), padding="same", name="pool4")(x)
        
        x = AveragePooling2D((4, 4), name="pool5")(x)
        x = Dropout(0.4, name='dropout')(x)
        
        x = Flatten(name='flatten')(x)
        x = Dense(classes, name='label')(x)
        x = Activation('softmax', name='softmax')(x)
        
        model = Model(inputs, x, name='googlenet')
        return model