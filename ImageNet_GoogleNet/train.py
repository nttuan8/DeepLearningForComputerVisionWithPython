# -*- coding: utf-8 -*-
"""
Created on Fri Jan 11 16:03:04 2019

@author: DELL
"""

import config
from Preprocessing.meanProcessor import MeanProcessor
from Preprocessing.simpleProcessor import SimplePreprocessor
from Preprocessing.imageToArrayProcessor import ImageToArrayProcessor

from Utils.hdf5DatasetGenerator import HDF5DatasetGenerator

from deeperGoogleNet import DeeperGoogleNet
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam

import json

aug = ImageDataGenerator(rotation_range=20, zoom_range=0.15, width_shift_range=0.2, height_shift_range=0.2,
                         shear_range=0.15, horizontal_flip=True, fill_mode='nearest')
mean = json.loads(open(config.DATASET_MEAN).read())

# Resize image, use for validation data
sp = SimplePreprocessor(64, 64)
mp = MeanProcessor(mean['R'], mean['G'], mean['B'])
iap = ImageToArrayProcessor()

trainGen = HDF5DatasetGenerator(config.TRAIN_HDF5, 64, aug=aug, preprocessors=[sp, mp, iap], classes=config.NUM_CLASSES)
valGen = HDF5DatasetGenerator(config.VAL_HDF5, 64, aug=aug, preprocessors=[sp, mp, iap], classes=config.NUM_CLASSES)

opt = Adam(1e-3)
model = DeeperGoogleNet.build(64, 64, 3, config.NUM_CLASSES, 0.0002)
model.compile(opt, 'categorical_crossentropy', ['accuracy'])

H = model.fit_generator(trainGen.generator(),
                    steps_per_epoch=trainGen.numImages//64,
                    validation_data=valGen.generator(),
                    validation_steps=valGen.numImages//64,
                    epochs=10,
                    max_queue_size=64*2)

model.save(config.MODEL_PATH, overwrite=True)

trainGen.close()
valGen.close()

X = trainGen.generator()
for x in X:
    print(x[1].shape)