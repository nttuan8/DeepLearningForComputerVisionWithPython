# -*- coding: utf-8 -*-
"""
Created on Fri Jan  4 21:43:03 2019

@author: DELL
"""

from keras.preprocessing.image import img_to_array

class ImageToArrayProcessor:
    def __init__(self, dataFormat=None):
        #store image data format
        self.dataFormat = dataFormat
        
    def preprocess(self, image):
        return img_to_array(image, data_format=self.data_format)