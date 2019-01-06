# -*- coding: utf-8 -*-
"""
Created on Sun Jan  6 17:22:50 2019

@author: DELL
"""
from utils.captchaHelper import preprocessing
from keras.models import load_model
import imutils
import cv2

model = load_model('captcha_lenet.hdf5')

imagePath = 'Download/1.jpg'
counts = {}
result = ''

image = cv2.imread(imagePath)
gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
gray = cv2.copyMakeBorder(gray, 8, 8, 8, 8, cv2.BORDER_REPLICATE)
thres = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
        
#contour
cnts = cv2.findContours(thres.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if imutils.is_cv2() else cnts[1]
cnts = cnts[::-1]
#cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:4]

for c in cnts:
   (x, y, w, h) = cv2.boundingRect(c)
   roi = gray[y-5:y+h+5, x-5:x+w+5]
   roi = preprocessing(roi, 28, 28)
   roi = roi.reshape(1, 28, 28, 1)
     
   prediction = model.predict(roi)
   result += str(prediction.argmax(1) + 1)

cv2.imshow('captcha', image)
print(result)