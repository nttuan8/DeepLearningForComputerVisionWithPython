# -*- coding: utf-8 -*-
"""
Created on Sun Jan  6 16:33:37 2019

@author: DELL
"""

import imutils
import cv2

def preprocessing(image, width, height):
    (h, w) = image.shape[:2]
    
    # resize image to larger dimension
    if w > h:
        image = imutils.resize(image, width=width)
    else:
        image = imutils.resize(image, height=height)
        
    # pad other dimension
    padW = int((width - image.shape[1])/2)
    padH = int((height - image.shape[0])/2)
    
    image = cv2.copyMakeBorder(image, padH, padH, padW, padW, cv2.BORDER_REPLICATE)
    image = cv2.resize(image, (width, height))
    
    return image