# -*- coding: utf-8 -*-
"""
Created on Sat Jan 12 09:02:21 2019

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
from keras.regularizers import l2
from keras.layers import Input
from keras.layers import add

from keras.models import Model

class ResNet:
    @staticmethod
    def residualModule(data, K, stride, axis, reduce=False, reg=0.0001, bnEpsilon=2e-5, bnMom=0.9):
        shortcut = data
        bn1 = BatchNormalization(axis=axis, epsilon=bnEpsilon, momentum=bnMom)(data)
        ac1 = Activation('relu')(bn1)
        
        if reduce:
            shortcut = Conv2D(K, (1,1), strides=stride, use_bias=False, kernel_regularizer=l2(reg))(ac1)
        else:
            stride = (1,1)
        
        conv1 = Conv2D(int(K*0.25), (1,1), use_bias=False, kernel_regularizer=l2(reg))(ac1)
        
        bn2 = BatchNormalization(axis=axis, epsilon=bnEpsilon, momentum=bnMom)(conv1)
        ac2 = Activation('relu')(bn2)
        conv2 = Conv2D(int(K*0.25), (3,3), strides=stride, padding='same', 
                       use_bias=False, kernel_regularizer=l2(reg))(ac2)
        
        bn3 = BatchNormalization(axis=axis, epsilon=bnEpsilon, momentum=bnMom)(conv2)
        ac3 = Activation('relu')(bn3)
        conv3 = Conv2D(K, (1,1), use_bias=False, kernel_regularizer=l2(reg))(ac3)
        
        
        x = add([conv3, shortcut])
        return x
    
    # stages=(3, 4, 6) and filters=(64, 128,256, 512) Conv(filters[0]) -> stage[i] * Residual(filters[i+1]) -> ...
    @staticmethod
    def build(width, height, depth, classes, stages, filters,
              reg=0.0001, bnEpsilon=2e-5, bnMom=0.9, dataset='cifar'):
        
        input_shape=(width, height, depth)
        axis = -1
        
        inputs = Input(shape=input_shape)
        x = BatchNormalization(axis=axis, epsilon=bnEpsilon, momentum=bnMom)(inputs)
        
        if dataset == 'cifar':
            x = Conv2D(filters[0], (3,3), use_bias=True, padding='same', kernel_regularizer=l2(reg))(x)
            for i in range(0, len(stages)):
                strides = (1,1) if i==0 else (2,2)
                x = ResNet.residualModule(x, filters[i+1], strides, axis, True, reg, bnEpsilon, bnMom)
                
                for j in range(0, stages[i]-1):
                    x = ResNet.residualModule(x, filters[i+1], (1,1), axis, False, reg, bnEpsilon, bnMom)
                
        x = BatchNormalization(axis=axis, epsilon=bnEpsilon, momentum=bnMom)(x)
        x = Activation('relu')(x)
        x = AveragePooling2D((8,8))(x)
        
        x = Flatten()(x)
        x = Dense(classes, kernel_regularizer=l2(reg), activation='softmax')(x)
        
        model = Model(inputs, x, name='resnet')
        return model
        