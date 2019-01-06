# -*- coding: utf-8 -*-
"""
Created on Sun Jan  6 14:22:47 2019

@author: DELL
"""

from imutils import paths
import imutils
import cv2
import os

imagePaths = list(paths.list_images('Download'))
counts = {}

for (i, imagePath) in enumerate(imagePaths):
    print('[INFO] Process image {}/{}'.format(i, len(imagePaths)))
    
    try:
        image = cv2.imread(imagePath)
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        gray = cv2.copyMakeBorder(gray, 8, 8, 8, 8, cv2.BORDER_REPLICATE)
        thres = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
        
        #contour
        cnts = cv2.findContours(thres.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if imutils.is_cv2() else cnts[1]
        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:4]
        for c in cnts:
            (x, y, w, h) = cv2.boundingRect(c)
            roi = gray[y-5:y+h+5, x-5:x+w+5]
            cv2.imshow('ROI', imutils.resize(roi, width=28))
            key = cv2.waitKey(0)
            
            key = chr(key).upper()
            dirPath = os.path.join('Dataset', key)
            if not os.path.exists(dirPath):
                os.makedirs(dirPath)
            
            count = counts.get(key, 1)
            p = os.path.sep.join([dirPath, '{}.png'.format(str(count).zfill(6))])
            cv2.imwrite(p, roi)
            
            counts[key] = count+1
    except KeyboardInterrupt:
        print('[INFO] Manually leave script')
        break
            