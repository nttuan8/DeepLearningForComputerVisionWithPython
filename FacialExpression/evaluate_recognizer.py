# -*- coding: utf-8 -*-
"""
Created on Sun Jan 13 10:08:24 2019

@author: DELL
"""

from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator

import config
from Preprocessing.imageToArrayProcessor import ImageToArrayProcessor
from Utils.hdf5DatasetGenerator import HDF5DatasetGenerator

valAug = ImageDataGenerator(rescale=1/255.0)

iap = ImageToArrayProcessor()

testGen = HDF5DatasetGenerator(config.VAL_HDF5, config.BATCH_SIZE, [iap], valAug, True, config.NUM_CLASSES)

model = load_model('')

loss, acc = model.evaluate_generator(testGen.generator(), testGen.numImages//config.BATCH_SIZE,
                                     max_queue_size=2*config.BATCH_SIZE)

print('Accuracy {}'.format(acc))

testGen.close()