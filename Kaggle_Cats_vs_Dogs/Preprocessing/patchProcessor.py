# -*- coding: utf-8 -*-
"""
Created on Wed Jan  9 22:08:01 2019

@author: DELL

This class randomly crop image to derised size, ex 256*256 -> 227*227
One kind of data augmentation and reduce overfitting

"""

from sklearn.feature_extraction.image import extract_patches_2d

class PatchProcessor:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        
    def preprocess(self, image):
        return extract_patches_2d(image, (self.height, self.width), max_patches=1)[0]