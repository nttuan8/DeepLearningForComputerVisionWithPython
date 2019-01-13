# -*- coding: utf-8 -*-
"""
Created on Sat Jan  5 17:21:39 2019

@author: DELL
"""

from keras.callbacks import BaseLogger
import matplotlib.pyplot as plt
import numpy as np
import json
import os

class TrainingMonitor(BaseLogger):
    def __init__(self, figPath, jsonPath=None, startAt=0):
        super(TrainingMonitor, self).__init__()
        self.figPath = figPath
        self.jsonPath = jsonPath
        self.startAt = startAt
        
    def on_train_begin(self, log={}):
        self.H = {}
        
        if self.jsonPath is not None:
            if os.path.exists(self.jsonPath):
                self.H = json.loads(open(self.jsonPath).read())
                if self.startAt > 0:
                    for k in self.H.keys:
                        self.H[k] = self.H[k][:self.startAt]
                        
    def on_epoch_end(self, epoch, logs={}):
        for (k,v) in logs.items():
            l = self.H.get(k, [])
            l.append(v)
            self.H[k] = l
            
        if self.jsonPath is not None:
            f = open(self.jsonPath, 'w')
            f.write(json.dumps(self.H))
            f.close()
        plt.ioff()
        if len(self.H['loss']) > 1:
            numOfEpoch = len(self.H['loss'])
            plt.plot(np.arange(0, numOfEpoch), self.H['loss'], label='training loss')
            plt.plot(np.arange(0, numOfEpoch), self.H['val_loss'], label='validation loss')
            plt.plot(np.arange(0, numOfEpoch), self.H['acc'], label='accuracy')
            plt.plot(np.arange(0, numOfEpoch), self.H['val_acc'], label='validation accuracy')
            plt.title('Accuracy and Loss epoch [{}]'.format(epoch))
            plt.xlabel('Epoch')
            plt.ylabel('Loss|Accuracy')
            plt.legend()
            
            #figPath = os.path.sep.join([self.figPath, '{}.png'.format(epoch)])
            plt.savefig(self.figPath)
            plt.close()