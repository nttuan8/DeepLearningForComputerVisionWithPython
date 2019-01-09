# -*- coding: utf-8 -*-
"""
Created on Wed Jan  9 14:37:35 2019

@author: DELL
"""

from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from Preprocessing.SimpleProcessor import SimplePreprocessor
from Preprocessing.AspectAwareProcessor import AspectAwareProcessor
from Preprocessing.ImageToArrayProcessor import ImageToArrayProcessor
from Dataset.SimpleDatasetLoader import SimpleDatasetLoader

from keras.optimizers import SGD
from keras.optimizers import RMSprop
from keras.applications import VGG16
from keras.layers import Input
from keras.models import Model
from keras.preprocessing.image import ImageDataGenerator

from imutils import paths

from NeurualNetwork.ConvolutionNeuralNetwork.fcHeadNet import FcHeadNet

aug = ImageDataGenerator(rotation_range=30, width_shift_range=0.1, height_shift_range=0.1, shear_range=0.2, 
                         zoom_range=0.2, horizontal_flip=True, fill_mode='nearest')

imagePaths = list(paths.list_images("Dataset/oxfordflower17/jpg"))
print("[INFO]Load images")

aap = AspectAwareProcessor(224, 224)
imgToArr = ImageToArrayProcessor()
loader = SimpleDatasetLoader(processors=[aap, imgToArr])
data, label = loader.load(imagePaths, verbose=500)
data = data.astype('float')/255.

X_train, X_test, y_train, y_test = train_test_split(data, label, test_size=0.25, random_state=42)

lb = LabelBinarizer()
y_train = lb.fit_transform(y_train)
y_test = lb.fit_transform(y_test)

#baseModel = VGG16(weights='imagenet', include_top=False, input_shape=Input(shape=(224, 224, 3)))
baseModel = VGG16(weights="imagenet", include_top=False, input_tensor=Input(shape=(224, 224, 3)))
headModel = FcHeadNet.build(baseModel, 17, 256)

model = Model(inputs=baseModel.input, outputs=headModel)
# freeze VGG model
for layer in baseModel.layers:
    layer.trainable = False

opt = RMSprop(0.001)
model.compile(opt, 'categorical_crossentropy', ['accuracy'])
numOfEpoch = 25
H = model.fit_generator(aug.flow(X_train, y_train, batch_size=32), validation_data=(X_test, y_test), steps_per_epoch=len(X_train)//32, epochs=numOfEpoch)

prediction = model.predict(X_test, batch_size=32)
print(classification_report(y_test.argmax(1), prediction.argmax(1)))

# unfreeze some last CNN layer:
for layer in baseModel.layers[15:]:
    layer.trainable = True

numOfEpoch = 100
opt = SGD(0.001)
model.compile(opt, 'categorical_crossentropy', ['accuracy'])
H = model.fit_generator(aug.flow(X_train, y_train, batch_size=32), validation_data=(X_test, y_test), steps_per_epoch=len(X_train)//32, epochs=numOfEpoch)