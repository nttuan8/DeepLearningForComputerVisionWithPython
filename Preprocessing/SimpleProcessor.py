# -*- coding: utf-8 -*-
"""
Created on Wed Dec 19 08:47:56 2018

@author: DELL
"""

import cv2

class SimplePreprocessor:
    def __init__(self, width, height, inter=cv2.INTER_AREA):
        self.width = width
        self.heigth = height
        self.inter = inter
    
    def preprocess(self, image):
        return cv2.resize(image, (self.width, self.heigth), interpolation=self.inter)
    
    
def add(a,b):
    return a+b

class Test:
    def __init__(self, a, b):
        self.a = a
        self.b = b
    def add(self):
        return self.a+self.b