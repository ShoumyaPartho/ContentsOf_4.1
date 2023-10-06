# -*- coding: utf-8 -*-
"""
Created on Tue May  2 11:51:26 2023

@author: NLP Lab
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt


def equalization(img):
    L = 256
    height, width = img.shape
    freq = cv2.calcHist([img], [0], None, [256], [0, 255])
    pdf = freq / (height * width)
    cdf = np.zeros(256, dtype=np.float32)
    cdf[0] = pdf[0]
    s = np.zeros(256, dtype=np.float32)
    for i in range(1, 256):
        cdf[i] = cdf[i - 1] + pdf[i]
        s[i] = round((L - 1) * cdf[i])

    output = np.zeros_like(img)

    for i in range(height):
        for j in range(width):
            x = img[i][j]
            output[i][j] = s[x]
    return output


img = cv2.imread("color_img.jpg")
cv2.imshow("input", img)

plt.subplot(2, 3, 1)
plt.title("Input Image Histogram")
plt.hist(img.ravel(), 256, [0, 255])
output1 = img.copy()

for i in range(3):
    s = equalization(img[:, :, i])
    output1[:, :, i] = s

cv2.imshow("RGB Output", output1)
cv2.imwrite("RGBOUTPUT.jpg", output1)

plt.subplot(2, 3, 2)
plt.title("Output Red Channel Histogram")
plt.hist(output1[:, :, 0].ravel(), 256, [0, 255], color="r")

plt.subplot(2, 3, 3)
plt.title("Output Green Channel Histogram")
plt.hist(output1[:, :, 1].ravel(), 256, [0, 255], color="g")

plt.subplot(2, 3, 4)
plt.title("Output Blue Channel Histogram")
plt.hist(output1[:, :, 2].ravel(), 256, [0, 255], color="b")


hsv_img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
plt.subplot(2, 3, 5)
plt.title("Input HSV")
plt.hist(hsv_img[:, :, 2].ravel(), 256, [0, 255])

hsv_img[:, :, 2] = equalization(hsv_img[:, :, 2])
hsv2rgb = cv2.cvtColor(hsv_img, cv2.COLOR_HSV2RGB)
plt.subplot(2, 3, 6)
plt.title("HSV Output")
plt.hist(hsv2rgb[:, :, 2].ravel(), 256, [0, 255])
plt.show()
cv2.imshow("HSV Output", hsv2rgb)
# cv2.imwrite("HSVOUTPUT.jpg", hsv2rgb)


cv2.waitKey(0)
cv2.destroyAllWindows()
