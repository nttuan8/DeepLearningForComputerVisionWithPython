# -*- coding: utf-8 -*-
"""
Created on Sat Jan 12 22:10:38 2019

@author: DELL
"""

import matplotlib.pyplot as plt
import numpy as np
import os
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint

import config
from Preprocessing.imageToArrayProcessor import ImageToArrayProcessor
from Utils.trainingMonitor import TrainingMonitor
from Utils.hdf5DatasetGenerator import HDF5DatasetGenerator
from emotion_VGG_model import VGGModelEmotion

trainAug = ImageDataGenerator(rotation_range=10, zoom_range=0.1, horizontal_flip=True, rescale=1/255.0,
                              fill_mode='nearest')
valAug = ImageDataGenerator(rescale=1/255.0)

iap = ImageToArrayProcessor()

trainGen = HDF5DatasetGenerator(config.TRAIN_HDF5, config.BATCH_SIZE, [iap], trainAug, True, config.NUM_CLASSES)
valGen = HDF5DatasetGenerator(config.VAL_HDF5, config.BATCH_SIZE, [iap], valAug, True, config.NUM_CLASSES)

model = VGGModelEmotion.build(48, 48, 1, config.NUM_CLASSES)
opt = Adam(1e-3)
model.compile(opt, 'categorical_crossentropy', ['accuracy'])

fname = os.path.sep.join([config.OUTPUT_PATH, "model-{epoch:03d}.hdf5"])
checkPoint = ModelCheckpoint(fname, monitor='val_loss', mode='min', save_best_only=True, verbose=1)

folPath = os.path.sep.join([config.OUTPUT_PATH, '{}.png'.format(os.getpid())])
jsonPath = os.path.sep.join([config.OUTPUT_PATH, '{}.json'.format(os.getpid())])

callbacks = [TrainingMonitor(folPath, jsonPath), checkPoint]

H = model.fit_generator(trainGen.generator(), steps_per_epoch=trainGen.numImages//config.BATCH_SIZE,
                    validation_data=valGen.generator(),validation_steps=valGen.numImages//config.BATCH_SIZE,
                    epochs=15,
                    max_queue_size=config.BATCH_SIZE*2,
                    callbacks=callbacks, verbose=1)
trainGen.close()
valGen.close()