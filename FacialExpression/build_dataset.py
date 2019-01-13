# -*- coding: utf-8 -*-
"""
Created on Sat Jan 12 21:14:26 2019

@author: DELL
"""

import config
from Utils.hdf5DatasetWriter import HDF5DatasetWriter
import numpy as np

f = open(config.INPUT_PATH)
# skip the header of csv file
f.__next__()

(trainImages, trainLabels) = ([], [])
(valImages, valLabels) = ([], [])
(testImages, testLabels) = ([], [])

for row in f:
    (label, image, usage) = row.strip().split(',')
    
    label = int(label)
    if label == 1:
        label = 0
    if label > 0:
        label -= 1
    
    image = np.array(image.split(' '), dtype='uint8')
    image = image.reshape((48, 48))
    
    if usage == 'Training':
        trainImages.append(image)
        trainLabels.append(label)
    elif usage == 'PrivateTest':
        valImages.append(image)
        valLabels.append(label)
    else:
        testImages.append(image)
        testLabels.append(label)

dataset = [[config.TRAIN_HDF5, trainImages, trainLabels],
           [config.VAL_HDF5, valImages, valLabels],
           [config.TEST_HDF5, testImages, testLabels]]

for (outputPath, images, labels) in dataset:
    writer = HDF5DatasetWriter((len(images), 48, 48), outputPath)
    for (image, label) in zip(images, labels):
        writer.add([image], [label])
    writer.close()

f.close()

import numpy as np
log = {'train':[1,2,3], 'test':[5,6,7]}
for k in log.keys():
    print(k, log[k])