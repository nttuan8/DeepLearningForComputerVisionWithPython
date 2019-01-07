# -*- coding: utf-8 -*-
"""
Created on Mon Jan  7 08:57:36 2019

@author: DELL
"""

from keras.preprocessing.image import img_to_array
from keras.models import load_model
import numpy as np
import imutils
import cv2
import time

face_classifier = cv2.CascadeClassifier('frontalface.xml')
model = load_model('miniVGG_emotion.hdf5')

emotions = {0: 'anger', 1: 'disgust', 2: 'fear', 3: 'happy', 4: 'sad', 5: 'surprise', 6:'neutral'}

cap = cv2.VideoCapture(0)

while(1):
    # Capture frame-by-frame
    ret, frame = cap.read()
    print(ret)
    if ret:
        frame = imutils.resize(frame, width=300)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frameClone = frame.copy()
        
        rects = face_classifier.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)
        for (fX, fY, fW, fH) in rects:
            roi = gray[fY:fY+fH, fX:fX+fW]
            roi = cv2.resize(roi, (48, 48))
            roi = roi.astype('float')/255.
            roi = img_to_array(roi)
            roi = np.expand_dims(roi, axis=0)
            
            predict = model.predict(roi)
            predict = predict.argmax(1)
            label = emotions.get(predict[0])
            cv2.putText(frameClone, label, (fX, fY-10), cv2.FONT_HERSHEY_COMPLEX, 0.45, (0, 0, 255), 2)
            cv2.rectangle(frameClone, (fX, fY), (fX+fW, fY+fH), (0, 0, 255), 2)
        cv2.imshow('Face', frameClone)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        cap.release()
        cv2.destroyAllWindows()

cap.release()
cv2.destroyAllWindows()