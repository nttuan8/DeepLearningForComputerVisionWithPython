# -*- coding: utf-8 -*-
"""
Created on Mon Jan  7 17:10:06 2019

@author: DELL
"""

import imutils
import cv2

class AspectAwareProcessor:
    def __init__(self, width, height, inter = cv2.INTER_AREA):
        self.width = width
        self.height = height
        self.inter = inter
        
    # Resize image keeping the ratio
    def preprocess(self, image):
        (h, w) = image.shape[:2]
        dW = 0
        dH = 0
        
        if h > w:
            image = imutils.resize(image, width=self.width)
            dH = int((image.shape[0] - self.height)/2.)
        else:
            image = imutils.resize(image, height=self.height)
            dW = int((image.shape[1] - self.width)/2.)
        
        if(dH < 0):
            image = cv2.copyMakeBorder(image, -dH, -dH, dW, dW, cv2.BORDER_REPLICATE)
        elif dW < 0:
            image = cv2.copyMakeBorder(image, dH, dH, -dW, -dW, cv2.BORDER_REPLICATE)
        else:
            image = image[dH:dH+self.height, dW:dW+self.width]
        return cv2.resize(image, (self.width, self.height), interpolation = self.inter)