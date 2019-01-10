# -*- coding: utf-8 -*-
"""
Created on Tue Jan  8 21:24:45 2019

@author: DELL
"""

import h5py
import os

class HDF5DatasetWriter:
    def __init__(self, dims, outputPath, dataKey='image', buffSize=1000):
        if os.path.exists(outputPath):
            raise ValueError('Output path exists, please manually delete it')
        
        self.db = h5py.File(outputPath, 'w')
        self.data = self.db.create_dataset(dataKey, dims, dtype='float')
        self.label = self.db.create_dataset('labels', (dims[0], ), dtype='int')
        
        self.bufSize = buffSize
        self.buffer = {'data':[], 'label':[]}
        self.idx = 0
    
    def add(self, row, label):
        self.buffer['data'].extend(row)
        self.buffer['label'].extend(label)
        
        if(len(self.buffer['data']) > self.bufSize):
            self.fflush()
            
    def fflush(self):
        i = self.idx + len(self.buffer['data'])
        self.data[self.idx: i] = self.buffer['data']
        self.label[self.idx: i] = self.buffer['label']
        self.idx = i
        self.buffer = {'data':[], 'label':[]}
        
    def storeClassLabel(self, labelClasses):
        dt = h5py.special_dtype(vlen=str)
        labelSet = self.db.create_dataset('labelName', (len(labelClasses), ), dtype=dt)
        labelSet[:] = labelClasses
        
    def close(self):
        if len(self.buffer['data']) > 0:
            self.fflush()
        self.db.close()