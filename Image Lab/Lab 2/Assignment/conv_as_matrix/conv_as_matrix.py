# -*- coding: utf-8 -*-
"""
Created on Tue June 27 05:42:59 2023

@author: SHOUMYA
"""

import numpy as np
import cv2 as cv

def toeplitz(c,l):
    s = np.zeros((len(c),l))
    
    for i in range(s.shape[0]):
        for j in range(s.shape[1]):
              s[j][i] = c[j]
        break
    
              
    for i in range(1,s.shape[0]):
        for j in range(i,s.shape[1]):
              s[j][i] = s[j-1][i-1]
            
    return s

def conv(img,filt):
    S = img.shape
    F = filt.shape

    R = S[0] + F[0] -1
    C = S[1] + F[1] -1

    Z = np.zeros((R,C))


    for x in range(S[0]):
        for y in range(S[1]):
            Z[x+int((F[0]-1)/2), y+int((F[1]-1)/2)] = img[x,y]


    for i in range(S[0]):
        for j in range(S[1]):
            k = Z[i:i+F[0],j:j+F[1]]
            sum = 0
            for x in range(F[0]):
                for y in range (F[1]):
                        sum += (k[x][y] * filt[-x][-y])

            img[i,j] = sum
    return img

img = cv.imread("Lena.jpg",cv.IMREAD_GRAYSCALE)
#img = cv.resize(img, (150,150))

cv.imshow("original",img)

kernel = np.array(([1,2,1],[2,4,2],[1,2,1]))/16

R = img.shape[0] + kernel.shape[0] -1
C = img.shape[1] + kernel.shape[1] -1

padded_kernel = np.pad(kernel,((R-kernel.shape[0],0),(0,C-kernel.shape[1])),'constant',constant_values=0)

#print(padded_kernel)
#print(padded_kernel.shape)

tp_list = []


for x in range(R-1,-1,-1):
    c = padded_kernel[x,:]
    #r = np.r_[c[0],np.zeros(img.shape[0]-1)]
    
    tp_item = toeplitz(c,img.shape[1])
    tp_list.append(tp_item)

#print(tp_list)

c = range(1,R+1)
#r = np.r_[c[0],np.zeros(img.shape[0]-1,dtype=int)]
db_index = toeplitz(c,img.shape[0])

db_index = np.array(db_index,dtype='int')

sp = tp_list[0].shape

b_h = sp[0]
b_w = sp[1]

h = sp[0] * db_index.shape[0]
w = sp[1] * db_index.shape[1]

db_blocked = np.zeros([h,w])

for i in range(db_index.shape[0]):
    for j in range(db_index.shape[1]):
        
        s_i = i * b_h
        s_j = j * b_w
        e_i = s_i + b_h
        e_j = s_j + b_w
        
        db_blocked[s_i:e_i,s_j:e_j] = tp_list[db_index[i][j]-1]

v_i = []

for i in range(img.shape[0]-1,-1,-1):
    for j in range(img.shape[1]):
        v_i.append(img[i][j])
#print(db_blocked.shape,len(v_i))
#print(v_i)
result = np.matmul(db_blocked,v_i)

ans = np.zeros((R,C))

for i in range(R):
    s = i * C
    nx = s + C
    ans[-i,:] = result[s:nx]

#print(ans)
ans = cv.normalize(ans, None, 0, 255,cv.NORM_MINMAX,dtype=cv.CV_8U)
#print(ans)
cv.imshow("Convolution as toeplitz matrix",ans)
cv.imshow("Gaussian Convolution",conv(img, kernel))

cv.waitKey()
cv.destroyAllWindows()
    
    