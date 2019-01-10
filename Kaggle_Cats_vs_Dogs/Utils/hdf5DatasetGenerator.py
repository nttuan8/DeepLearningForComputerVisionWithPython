# -*- coding: utf-8 -*-
"""
Created on Tue Jan  8 21:24:45 2019

@author: DELL
"""

import h5py
import numpy as np
from keras.utils import np_utils

class HDF5DatasetGenerator:
    def __init__(self, dbPath, batch_size, preprocessors=None, aug=None, binarize=True, classes=2):
        self.batch_size = batch_size
        self.preprocessor = preprocessors
        self.aug = aug
        self.binarize = binarize
        self.classes = classes
        
        self.db = h5py.File(dbPath)
        self.numImages = self.db['labels'].shape[0]
        
    
    def generator(self, passes=np.inf):
        epochs = 0
        
        while epochs < passes:
            for i in np.arange(0, self.numImages, self.batch_size):
                images = self.db['image'][i:i+self.batch_size]
                labels = self.db['labels'][i:i+self.batch_size]
                if self.binarize:
                    labels = np_utils.to_categorical(labels, self.classes)
                if self.preprocessor is not None:
                    processedImage = []
                    for image in images:
                        for processor in self.preprocessor:
                            image = processor.preprocess(image)
                        processedImage.append(image)
                    images = np.array(processedImage)
                if self.aug is not None:
                    (images, labels) = next(self.aug.flow(images, labels, batch_size=self.batch_size))
                yield(images, labels)
                epochs += 1
    
    def close(self):
        self.db.close()