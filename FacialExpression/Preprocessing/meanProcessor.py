# -*- coding: utf-8 -*-
"""
Created on Wed Jan  9 22:01:53 2019

@author: DELL
"""
import cv2

class MeanProcessor:
    def __init__(self, rMean, gMean, bMean):
        self.rMean = rMean
        self.gMean = gMean
        self.bMean = bMean
        
    def preprocess(self, image):
        (B, G, R) = cv2.split(image.astype('float32'))
        R -= self.rMean
        G -= self.gMean
        B -= self.bMean
        return cv2.merge([B, G, R])