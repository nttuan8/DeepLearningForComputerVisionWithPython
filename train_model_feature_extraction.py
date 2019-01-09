# -*- coding: utf-8 -*-
"""
Created on Wed Jan  9 09:45:51 2019

@author: DELL
"""

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
import h5py

db = h5py.File('Dataset/oxfordflower17/hdf5/output.hdf5')
i = int(db['labels'].shape[0] * 0.75)

# Tuning parameter
params = {'C' : [0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0]}
model = GridSearchCV(LogisticRegression(), params, cv=3)
model.fit(db['features'][:i], db['labels'][:i])
print('Best parameter for the model {}'.format(model.best_params_))

# Evaluate
preds = model.predict(db["features"][i:])
print(classification_report(db["labels"][i:], preds))