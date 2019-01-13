# -*- coding: utf-8 -*-
"""
Created on Sat Jan 12 21:08:40 2019

@author: DELL
"""

from os import path

INPUT_PATH = 'Dataset/fer2013.csv'

# disgust emotion only has 113 images, compared to 1000 images in other emotions -> merge disgust and anger
NUM_CLASSES = 6

TRAIN_HDF5 = 'Dataset/Hdf5/train.hdf5'
TEST_HDF5 = 'Dataset/Hdf5/test.hdf5'
VAL_HDF5 = 'Dataset/Hdf5/val.hdf5'

BATCH_SIZE = 128
OUTPUT_PATH = 'Dataset/Output'