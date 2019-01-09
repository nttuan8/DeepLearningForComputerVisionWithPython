# -*- coding: utf-8 -*-
"""
Created on Wed Jan  9 14:23:31 2019

@author: DELL
"""

from keras.applications import VGG16

model = VGG16(weights='imagenet', include_top = True)
for(i, layer) in enumerate(model.layers):
    print('layer {} : {}'.format(i, layer.__class__.__name__))