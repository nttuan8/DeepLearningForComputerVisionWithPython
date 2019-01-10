# -*- coding: utf-8 -*-
"""
Created on Thu Jan 10 15:24:40 2019

@author: DELL
"""

import matplotlib.pyplot as plt
import numpy as np

import config
from Preprocessing.aspectAwareProcessor import AspectAwareProcessor
from Preprocessing.cropProcessor import CropProcessor
from Preprocessing.meanProcessor import MeanProcessor
from Preprocessing.patchProcessor import PatchProcessor
from Preprocessing.simpleProcessor import SimplePreprocessor
from Preprocessing.imageToArrayProcessor import ImageToArrayProcessor

from Utils.hdf5DatasetGenerator import HDF5DatasetGenerator

from alexNet import AlexNet

from sklearn.metrics import classification_report

from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam

import json

mean = json.loads(open(config.DATASET_MEAN).read())

# Resize image, use for validation data
sp = SimplePreprocessor(227, 227)
mp = MeanProcessor(mean['R'], mean['G'], mean['B'])
iap = ImageToArrayProcessor()

# Load model
model = load_model(config.MODEL_PATH)

testGen = HDF5DatasetGenerator(config.TEST_HDF5, 64, preprocessors=[sp, mp, iap], classes=2)
prediction = model.predict_generator(testGen.generator(), steps=testGen.numImages//64, max_queue_size=64*2)
print(classification_report(testGen.db['labels'], prediction))