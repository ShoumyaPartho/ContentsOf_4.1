# -*- coding: utf-8 -*-
"""
Created on Thu Jun 22 01:26:23 2023

@author: SHOUMYA

Tutorial Link: https://www.c-sharpcorner.com/article/log-and-inverse-log-transformation-on-image-in-python/
"""

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import math


img = cv.imread("einstein.jpg",cv.IMREAD_GRAYSCALE)

c = 255/np.log(256)

# Log Transform
out = c * np.log(1+img)
out = np.array(out,dtype = np.uint8)

# Inverse Log Transform
inv = np.exp(img/c)-1
inv = np.array(inv,dtype = np.uint8)

cv.imshow("Log", out)
cv.imshow("Inverse Log", inv)

cv.waitKey()
cv.destroyAllWindows()