# -*- coding: utf-8 -*-
"""
Created on Wed Dec 19 09:08:58 2018

@author: DELL
"""
import numpy as np
import cv2
import os

class SimpleDatasetLoader:
    def __init__(self, processors=None):
        self.processors = processors
        
        if self.processors is None:
            self.processors = []
    
    def load(self, imagePaths, verbose=-1):
        data = []
        labels = []
        
        for (i, imagePath) in enumerate(imagePaths):
            image = cv2.imread(imagePath)
            label = imagePath.split(os.path.sep)[-2]
            if self.processors is not None:
                for p in self.processors:
                    image = p.preprocess(image)
            data.append(image)
            labels.append(label)
            
            if verbose > 0 and i > 0 and (i+1)%verbose==0:
                print("Info process {}/{}".format(i+1, len(imagePaths)))
        return np.array(data), np.array(labels)