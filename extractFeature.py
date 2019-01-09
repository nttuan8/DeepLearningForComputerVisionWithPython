# -*- coding: utf-8 -*-
"""
Created on Tue Jan  8 21:48:29 2019

@author: DELL
"""

from keras.applications import VGG16
from keras.applications import imagenet_utils
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img

from sklearn.preprocessing import LabelEncoder
from IO.hdf5DatasetWriter import HDF5DatasetWrite
from imutils import paths
import numpy as np
import random
import os

image_path = list(paths.list_images('Dataset/train'))
random.shuffle(image_path)

labels = [p.split(os.path.sep)[-2] for p in image_path]
le = LabelEncoder()
labels = le.fit_transform(labels)

model = VGG16(weights='imagenet', include_top=False)

dataset = HDF5DatasetWrite(dims=(len(image_path), 512*7*7) , outputPath='output.hdf5', dataKey='features', buffSize=100)
dataset.storeClassLabel(le.classes_)

# batch_size=32
bs = 32
for i in np.arange(0, len(image_path), bs):
    batchPaths = image_path[i:i+bs]
    batchLabels = labels[i:i+bs]
    batchImage = []
    
    for (j, imagePath) in enumerate(batchPaths):
        image = load_img(imagePath, target_size=(224, 224))
        image = img_to_array(image)
        
        image = np.expand_dims(image, 0)
        image = imagenet_utils.preprocess_input(image)
        
        batchImage.append(image)
    
    batchImage = np.vstack(batchImage)
    features = model.predict(batchImage, batch_size=bs)
    features = features.reshape((features.shape[0], 512*7*7))
    dataset.add(features, batchLabels)
    print('batch {}'.format(i))

dataset.close()
#dataset.db.close()