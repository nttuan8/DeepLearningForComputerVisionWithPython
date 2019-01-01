# -*- coding: utf-8 -*-
"""
Created on Tue Jan  1 14:45:28 2019

@author: DELL
"""

from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from Preprocessing.SimpleProcessor import SimplePreprocessor
from Dataset.SimpleDatasetLoader import SimpleDatasetLoader

from imutils import paths
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True, help="Add path to dataset")
ap.add_argument("-k", "--neighbor", type=int, default=1, help="# of neighbors in knn")
ap.add_argument("-j", "--job", type=int, default=-1, help="# of jobs to run knn (default=1 requires all cores)")
args = vars(ap.parse_args())
                
imagePaths = list(paths.list_images(args['dataset']))
imagePaths = list(paths.list_images("F:/DeepLearning/DeepLearningForComputerVisionWithPython/Code/Dataset/train"))
print("[INFO]Load images")

processor = SimplePreprocessor(32, 32)
loader = SimpleDatasetLoader(processors=[processor])
data, label = loader.load(imagePaths, verbose=500)
data = data.reshape(data.shape[0], 3*32*32)

#Show memory comsumption
print("[INFO] Number of data comsumption {:.1f}MB".format(data.nbytes/(1024*1000.0)))

#encode the label cat,dog -> 0,1
le = LabelEncoder()
label = le.fit_transform(label)

#split train,test set
X_train, X_test, y_train, y_test = train_test_split(data, label, test_size=0.25, random_state=42)

#train the model
model = KNeighborsClassifier(n_neighbors=args['neighbor'], n_jobs=args['job'])
model.fit(X_train, y_train)

#evaluate
print(classification_report(y_test, model.predict(X_test), target_names=le.classes_))


