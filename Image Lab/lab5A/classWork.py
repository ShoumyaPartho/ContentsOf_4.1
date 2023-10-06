# -*- coding: utf-8 -*-
"""
Created on Tue May 30 11:33:25 2023

@author: SHOUMYA
"""

import cv2
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

def getInsect(kn,cimg):
    h,w = kn.shape
    n = np.zeros((h,w))
    
    for i in range(h):
        for j in range(w):
            if (kn[i,j] == 0):
                n[i,j] = kn[i,j]
            else:
                n[i,j] = cimg[i,j]
    return n

def isEqual(ko,kn):
    h,w = kn.shape
    
    for i in range(h):
        for j in range(w):
            if kn[i][j] != ko[i][j]:
                return 0
            
    return 1

def getUnion(kn,img):
    h,w = kn.shape
    n = np.zeros((h,w))
    
    for i in range(h):
        for j in range(w):
            n[i][j] = max(kn[i][j],img[i][j])
    return n


img = cv2.imread('input.jpg', 0)
h,w = img.shape
nimg = np.zeros((h,w))
cimg = np.zeros((h,w))
nimg[h//2][w//2] = 1
r, img = cv2.threshold(img, 130, 255, cv2.THRESH_BINARY)#_INV)
cv2.imshow("Original", img)
cv2.imwrite('H:/Academic/4-1/Image Lab/lab5A/Assignment/inp.jpg',img)
#cv2.imshow("Empty", nimg)

for i in range(h):
    for j in range(w):
        cimg[i][j] = 255 - img[i][j]

kernel = cv2.getStructuringElement(cv2.MORPH_CROSS,(5,5)) #cv2.MORPH_RECT for all 1s
print(kernel)
kernel = (kernel) *255
kernel = np.uint8(kernel)

rate = 50
kernel1 = cv2.resize(kernel, None, fx = rate, fy = rate, interpolation = cv2.INTER_NEAREST)
cv2.imshow("kernel",kernel1)
cv2.imwrite('H:/Academic/4-1/Image Lab/lab5A/Assignment/kernel.jpg',kernel1)

ko = nimg
kn = nimg
cnt = 0

while True:
    kn = cv2.dilate(ko,kernel,iterations = 1)
    kn = getInsect(kn,cimg)
    
    flag = isEqual(ko,kn)
    
    if flag==1:
        break
    
    ko = kn
    print(cnt+1)
    cnt +=1

kn = getUnion(kn,img)
    
cv2.imshow("Dilation", kn)
cv2.imwrite('H:/Academic/4-1/Image Lab/lab5A/Assignment/filled.jpg',kn)


cv2.waitKey(0)
cv2.destroyAllWindows()
