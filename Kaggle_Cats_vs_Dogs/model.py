# -*- coding: utf-8 -*-
"""
Created on Wed Jan  9 20:47:35 2019

@author: DELL
"""

import config
from Preprocessing.aspectAwareProcessor import AspectAwareProcessor
from Utils.hdf5DatasetWriter import HDF5DatasetWriter

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

from imutils import paths
import numpy as np
import matplotlib.pyplot as plt

import cv2
import json
import os

trainPaths = list(paths.list_images(config.IMAGES_PATH))
trainLabels = [p.split(os.path.sep)[1].split('.')[0] for p in trainPaths]

le = LabelEncoder()
trainLabels = le.fit_transform(trainLabels)

trainPaths, testPaths, trainLabels, testLabels = train_test_split(trainPaths, trainLabels,
                                                                  test_size=config.NUM_TEST_IMAGES, random_state=42)
trainPaths, valPaths, trainLabels, valLabels = train_test_split(trainPaths, trainLabels,
                                                                  test_size=config.NUM_TEST_IMAGES, random_state=42)

datasets = [#('train', trainPaths, trainLabels, config.TRAIN_HDF5),
            ('test', testPaths, testLabels, config.TEST_HDF5),
            ('val', valPaths, valLabels, config.VAL_HDF5)]

aap = AspectAwareProcessor(256, 256)

(R, G, B) = ([], [], [])

for (dType, paths, labels, outputPath) in datasets:
    writer = HDF5DatasetWriter((len(paths), 256, 256, 3), outputPath)
    for (i, (path, label)) in enumerate(zip(paths, labels)):
        image = cv2.imread(path)
        image = aap.preprocess(image)
        if dType=='train':
            (b, g, r) = cv2.mean(image)[:3]
            R.append(r)
            G.append(g)
            B.append(b)
        writer.add([image], [label])
        if i%1000==0:
            print(i)
    writer.close()
    
print('[INFO] Write mean to json file')
D = {'R': np.mean(R), 'G': np.mean(G), 'B': np.mean(B)}
f = open(config.DATASET_MEAN, 'w+')
f.write(json.dumps(D))
f.close()