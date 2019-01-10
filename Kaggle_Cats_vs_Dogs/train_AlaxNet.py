# -*- coding: utf-8 -*-
"""
Created on Thu Jan 10 09:47:52 2019

@author: DELL
"""

import matplotlib.pyplot as plt
import config
from Preprocessing.aspectAwareProcessor import AspectAwareProcessor
from Preprocessing.cropProcessor import CropProcessor
from Preprocessing.meanProcessor import MeanProcessor
from Preprocessing.patchProcessor import PatchProcessor
from Preprocessing.simpleProcessor import SimplePreprocessor
from Preprocessing.imageToArrayProcessor import ImageToArrayProcessor

from Utils.hdf5DatasetGenerator import HDF5DatasetGenerator

from alexNet import AlexNet
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam

import json

aug = ImageDataGenerator(rotation_range=20, zoom_range=0.15, width_shift_range=0.2, height_shift_range=0.2,
                         shear_range=0.15, horizontal_flip=True, fill_mode='nearest')
mean = json.loads(open(config.DATASET_MEAN).read())

# Resize image, use for validation data
sp = SimplePreprocessor(227, 227)

# Randomly crop image, use for training
pp = PatchProcessor(227, 227)
mp = MeanProcessor(mean['R'], mean['G'], mean['B'])
iap = ImageToArrayProcessor()

trainGen = HDF5DatasetGenerator(config.TRAIN_HDF5, 128, aug=aug, preprocessors=[pp, mp, iap], classes=2)
valGen = HDF5DatasetGenerator(config.VAL_HDF5, 128, aug=aug, preprocessors=[sp, mp, iap], classes=2)

opt = Adam(0.001)
model = AlexNet.build(227, 227, 3, 2, 0.0002)
model.compile(opt, 'binary_crossentropy', ['accuracy'])

model.fit_generator(trainGen.generator(),
                    steps_per_epoch=trainGen.numImages//128,
                    validation_data=valGen.generator(),
                    validation_steps=valGen.numImages//128,
                    epochs=100,
                    max_queue_size=128*2)

model.save(config.MODEL_PATH, overwrite=True)

trainGen.close()
valGen.close()
