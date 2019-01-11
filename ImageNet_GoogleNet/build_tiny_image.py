# -*- coding: utf-8 -*-
"""
Created on Fri Jan 11 13:59:15 2019

@author: DELL
"""

import config
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from Utils.hdf5DatasetWriter import HDF5DatasetWriter

from imutils import paths
import numpy as np
import json
import cv2
import os

trainPaths = list(paths.list_images(config.TRAIN_IMAGES))
trainLabels = [p.split(os.path.sep)[-3] for p in trainPaths]
le = LabelEncoder()
trainLabels = le.fit_transform(trainLabels)

trainPaths, testPaths, trainLabels, testLabels = train_test_split(trainPaths, trainLabels, 
                                                                  test_size=config.NUM_TEST_IMAGE, stratify=trainLabels,
                                                                  random_state=42)

M = open(config.VAL_MAPPING).read().strip().split('\n')
M = [r.split('\t')[:2] for r in M]
valPaths = [os.path.sep.join([config.VAL_IMAGES, m[0]]) for m in M]
valLabels = le.fit_transform([m[1] for m in M])

datasets = [('train', trainPaths, trainLabels, config.TRAIN_HDF5),
            ('test', testPaths, testLabels, config.TEST_HDF5),
            ('val', valPaths, valLabels, config.VAL_HDF5)]

(R, G, B) = ([], [], [])

for dType, paths, labels, path in datasets:
    print('[INFO] Building {}'.format(dType))
    writer = HDF5DatasetWriter((len(paths), 64, 64, 3), path)
    
    for (i, (imagePath, label)) in enumerate(zip(paths, labels)):
        image = cv2.imread(imagePath)
        if dType == 'train':
            b, g, r = cv2.mean(image)[:3]
            R.append(r)
            G.append(g)
            B.append(b)
        writer.add([image], [label])
        if i%1000 == 0:
            print(i)
    writer.close()
print('[INFO] Write json file')
D = {'R':np.mean(R), 'G':np.mean(G), 'B':np.mean(B)}
f = open(config.DATASET_MEAN, 'w')
f.write(json.dumps(D))
f.close()
    