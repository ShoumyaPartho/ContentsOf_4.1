import numpy as np
import cv2 as cv

originalImage = cv.imread("lena.jpg",cv.IMREAD_GRAYSCALE)
cv.imshow("original",originalImage)

kernel = np.array([[1,1,1],[1,1,1],[1,1,1]]) * (1/9)

imageShape = originalImage.shape
kernelShape = kernel.shape

row = imageShape[0] + kernelShape[0] - 1
col = imageShape[1] + kernelShape[1] - 1

outImage = np.zeros((row,col))

for i in range(imageShape[0]):
    for j in range(imageShape[1]):
        outImage[i + int((kernelShape[0]-1)/2) , j + int((kernelShape[1]-1)/2)] = originalImage[i,j]

print(outImage)

for i in range(imageShape[0]):
    for j in range(imageShape[1]):
        k = outImage[i:i+kernelShape[0], j: j+kernelShape[1]]
        sum = 0
        
        for x in range(kernelShape[0]):
            for y in range(kernelShape[1]):
                sum += k[x][y]*kernel[x][y]
                
        originalImage[i][j] = sum
        
cv.imshow('FinalOUTPUT',originalImage)


cv.waitKey()
cv.destroyAllWindows()