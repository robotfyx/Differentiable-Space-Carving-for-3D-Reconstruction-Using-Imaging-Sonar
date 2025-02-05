# -*- coding: utf-8 -*-
"""
Created on Wed Nov  1 10:19:34 2023

@author: Administrator
"""

import pickle
import cv2
import numpy as np
import os

path = 'data/14deg_simpleBoat/Data'

for pkl in os.listdir(path):
    with open(os.path.join(path, pkl), 'rb') as f:
        data = pickle.load(f)
        image = data['ImagingSonar']
        # image[image<0.2] = 0
        cv2.imshow('sonar image', image)
        key = cv2.waitKey(-1)
        if key == 32:
            continue#按空格下一张
        if key == ord('q'):
            break#按q退出
        if key == ord('a'):
            img = image.copy()
            img[:, 246:266] = 0
            cv2.imshow('img', img)
            cv2.waitKey(5000)
            cv2.destroyWindow('img')
            
            mask = np.zeros(image.shape, dtype=np.uint8)
            mask[:, 246:266] = 255
            img1 = cv2.inpaint(img, mask, 10, cv2.INPAINT_NS)
            img1[img1<1e-2] = 0
            cv2.imshow('img1', img1)
            cv2.waitKey(5000)
            cv2.destroyWindow('img1')
        if key == ord('s'):
            cv2.imwrite('images/image to restore.png', np.uint8((image*255.0).round()))
        if key == ord('p'):
            print(len(np.where(image>0.3)[0]))
            print(len(np.where(image>0.1)[0]))
            
cv2.destroyAllWindows()


