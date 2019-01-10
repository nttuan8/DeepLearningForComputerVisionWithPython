# -*- coding: utf-8 -*-
"""
Created on Wed Jan  9 20:41:18 2019

@author: DELL
"""
IMAGES_PATH = '../Dataset/cat_dog/train'

NUM_CLASS = 2
NUM_VAL_IMAGES = 1250 * NUM_CLASS
NUM_TEST_IMAGES = 1250 * NUM_CLASS

TRAIN_HDF5 = 'Hdf5/train_data.hdf5'
TEST_HDF5 = 'Hdf5/test_data.hdf5'
VAL_HDF5 = 'Hdf5/val_data.hdf5'

OUTPUT_PATH = 'Outputs'
MODEL_PATH = 'Outputs/AlexNet_Cats_vs_Dogs.model'

DATASET_MEAN = 'Outputs/mean.json'