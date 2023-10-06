import cv2 as cv
import numpy as np


img = cv.imread("Lena.jpg",cv.IMREAD_GRAYSCALE)

cv.imshow("Original",img)

mn = img.min()

mx = img.max()

op = img.copy()

#print(img.shape)

for i in range(img.shape[0]):
    for j in range(img.shape[1]):
            op[i][j] = ((img[i][j] - mn)/(mx-mn))*255

cv.imshow("fianl",op)
cv.waitKey()
cv.destroyAllWindows()