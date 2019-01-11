# -*- coding: utf-8 -*-
"""
Created on Fri Jan 11 13:52:25 2019

@author: DELL
"""

from os import path

TRAIN_IMAGES = '../Dataset/tiny-imagenet-200/train'
VAL_IMAGES = '../Dataset/tiny-imagenet-200/val/images'

VAL_MAPPING = '../Dataset/tiny-imagenet-200/val/val_annotations.txt'

WORDNET_IDS = '../Dataset/tiny-imagenet-200/wnids.txt'
WORD_LABELS = '../Dataset/tiny-imagenet-200/words.txt'

NUM_CLASSES = 200
NUM_TEST_IMAGE = 50 * NUM_CLASSES

TRAIN_HDF5 = '../Dataset/tiny-imagenet-200/hdf5/train.hdf5'
VAL_HDF5 = '../Dataset/tiny-imagenet-200/hdf5/val.hdf5'
TEST_HDF5 = '../Dataset/tiny-imagenet-200/hdf5/test.hdf5'

DATASET_MEAN = 'Outputs/mean.json'
MODEL_PATH = 'Outputs/model.hdf5'