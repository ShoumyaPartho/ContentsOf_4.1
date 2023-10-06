# -*- coding: utf-8 -*-
"""
Created on Mon Apr  3 20:00:54 2023

@author: Joty
"""
import cv2
import math
import numpy as np

# img=np.array([[1,2,3],[4,5,6]])
#kernel=np.array([[10,20],[30,40]])
kernel = np.array(([0,-1,0],
                    [-1,5,-5],
                    [0,-1,2]), np.float32)
img=cv2.imread("lena128.jpg", cv2.IMREAD_GRAYSCALE)
img_h=img.shape[0]
img_w=img.shape[1]
kernel_h=kernel.shape[0]
kernel_w=kernel.shape[1]
output_row=kernel.shape[0]+img.shape[0]-1
output_col=kernel.shape[1]+img.shape[1]-1
pad_t=img_h-1
pad_r=img_w-1
pad_filter=cv2.copyMakeBorder(kernel, pad_t,0,0,pad_r,cv2.BORDER_CONSTANT)
f=np.zeros((output_row,output_col, img_w))
for i in range(output_row):
    ftmp=np.zeros((output_col,img_w))    #pad_r+1=img_w
    for j in range(kernel_w):
        for l in range(img_w):    
            ftmp[j+l][l]=pad_filter[output_row-1-i][j]
            #print(ftmp[j+l][l])
    f[i]=ftmp

double_blocked=np.zeros((f.shape[1]*f.shape[0],img_h*f.shape[2]))

for t in range(img_h):
    for f_matrix_no in range(f.shape[0]-t):
        for i in range(f.shape[1]):
            for j in range(f.shape[2]):        
                    double_blocked[(f_matrix_no+t)*f.shape[1]+i][(f.shape[2]*t)+j]=f[f_matrix_no][i][j]


#print(double_blocked)

img_col_vector=np.zeros((img_h*img_w,1))

k=0
for i in range(img_h):
    for j in range(img_w):
        img_col_vector[k][0]=img[img_h-1-i][j]
        k=k+1

result_col=np.matmul(double_blocked,img_col_vector)

output=np.zeros((output_row,output_col))

k=0

for i in range(output_row):
    for j in range(output_col):
        output[output_row-1-i][j]=result_col[k]
        #print(output[output_row-1-i][j])
        k=k+1

print(output)

cv2.imwrite("convolution.jpg",output)
cv2.waitKey(0)
cv2.destroyAllWindows()