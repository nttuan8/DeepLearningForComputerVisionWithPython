# -*- coding: utf-8 -*-
"""
Created on Mon Jan  7 16:00:43 2019

@author: DELL
"""

from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img
import numpy as np

image = load_img('Dataset/augmentation/cat.jpg')
image = img_to_array(image)
image = np.expand_dims(image, 0)

aug = ImageDataGenerator(rotation_range=30, width_shift_range=0.1, height_shift_range=0.1,
                         shear_range=0.2, zoom_range=0.2, horizontal_flip=True, fill_mode='nearest')

imgGen = aug.flow(image, batch_size=1, save_to_dir='Dataset/augmentation', save_format='jpg')

total = 0
for image in imgGen:
    total += 1
    
    if total == 10:
        break
