# -*- coding: utf-8 -*-
"""
Created on Tue Jun 13 11:09:21 2023

@author: SHOUMYA
"""

import numpy as np
import cv2 as cv
import math

def epa(m,n):
    epac = np.zeros((m,n))
    xc = m//2
    yc = n//2
    d = min(m - xc, n-yc)
    
    for i in range(m):
        for j in range(n):
            dx = (i-xc)
            dy = (j-yc)
            dx *= dx
            dy *= dy
            r = math.sqrt(dx + dy)
            t = r/d
            
            if t>1.0:
                epac[i][j] = 0
            else:
                epac[i][j] = 1-(t*t)
    return epac

def bilateral(img,filt,sigma):
    w1 = filt.shape[0]
    h1 = filt.shape[1]
    w = w1//2
    h = h1//2
    img = img/255
    m = img.shape[0]
    n = img.shape[1]
    out = np.zeros((m,n),np.float32)
    pi = 3.1416
    
    for i in range(m):
        for j in range(n):
            rs , factor = 0.0 , 0.0
            range_domain = np.zeros((w1,h1),np.float32)
            
            for x in range(-w,w+1):
                for y in range(-h,h+1):
                    if (i-x)>=0 and (i-x<m) and (j-y)>=0 and (j-y)<n:
                        t = img[i][j] - img[i-x][j-y]
                        t = (math.exp(-(t*t)/(2*sigma*sigma)))/((math.sqrt(2*pi))*sigma)
                        tm = filt[x+w][y+h] * t 
                        range_domain[x+w][y+h] = filt[x+w][y+h] * t * img[i-x][j-y]
                        rs += range_domain[x+w][y+h]
                        factor += tm
                        
            rs = (rs/factor)
            out[i][j] = rs
            
    out = out * 255
    out = cv.normalize(out,None,0,1.0,cv.NORM_MINMAX, dtype = cv.CV_32F)
    
    return out
    
    
    

img = cv.imread("cube.png", cv.IMREAD_GRAYSCALE)
cv.imshow("original",img)

l = int(input("Enter dimension of kernel: "))
sigma = int(input("Enter value for sigma: "))
epan_filt = epa(l,l)
print(epan_filt)

output = bilateral(img,epan_filt,sigma)
cv.imshow("Final Epanechnikov Bilateral",output)