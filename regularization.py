# -*- coding: utf-8 -*-
"""
Created on Thu Jan  3 09:32:58 2019

@author: DELL
"""

from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

from Preprocessing.SimpleProcessor import SimplePreprocessor
from Dataset.SimpleDatasetLoader import SimpleDatasetLoader

from imutils import paths

imagePaths = list(paths.list_images("./Dataset/train"))
print("[INFO]Load images")

processor = SimplePreprocessor(32, 32)
loader = SimpleDatasetLoader(processors=[processor])
data, label = loader.load(imagePaths, verbose=500)
data = data.reshape(data.shape[0], 3*32*32)

#encode the label cat,dog -> 0,1
le = LabelEncoder()
label = le.fit_transform(label)

#split train,test set
X_train, X_test, y_train, y_test = train_test_split(data, label, test_size=0.25, random_state=42)

for r in [None, 'l1', 'l2']:
    print('[INFO] Train model with {} penalty'.format(r))
    
    #model with softmax loss function and 10 epoches
    model = SGDClassifier(loss='log', penalty=r, max_iter=10, learning_rate='constant', eta0=0.01, random_state=42)
    model.fit(X_train, y_train)
    
    #evaluate
    acc = model.score(X_test, y_test)
    print('[INFO] Penalty {} accuracy {:.1f}'.format(r, acc*100))
    
    