# -*- coding: utf-8 -*-
"""
Created on Wed Jan  9 22:13:02 2019

@author: DELL
"""

import numpy as np
import cv2

class CropProcessor:
    def __init__(self, width, height, horiz=True, inter=cv2.INTER_AREA):
        self.width = width
        self.height = height
        self.horiz = horiz
        self.inter = inter
        
    def preprocess(self, image):
        crops = []
        (h, w) = image.shape[:2]
        coords = [[0, 0, self.width, self.height],
                  [w - self.width, 0, w, self.height],
                  [0, h - self.height, self.width, h],
                  [w - self.width, h - self.height, w, h]]
        dW = int((w - self.width)/2)
        dH = int((h - self.height)/2)
        coords.append([dW, dH, dW + self.width, dH + self.height])
        
        for (x, y, dx, dy) in coords:
            crop = image[y:dy, x:dx]
            crop = cv2.resize(crop, (self.width, self.height), interpolatioin=self.inter)
            crops.append(crop)
            
        if self.horiz:
            mirrors = [cv2.flip(c, 1) for c in crops]
            crops.extend(mirrors)
            
        return np.array(crops)