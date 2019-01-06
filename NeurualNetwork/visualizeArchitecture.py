# -*- coding: utf-8 -*-
"""
Created on Sun Jan  6 10:06:42 2019

@author: DELL
"""

from ConvolutionNeuralNetwork.lenet import LeNet
from keras.utils import plot_model

model = LeNet.build(32, 32, 1, 10)
plot_model(model, 'leNet.jpg', show_shapes=True)