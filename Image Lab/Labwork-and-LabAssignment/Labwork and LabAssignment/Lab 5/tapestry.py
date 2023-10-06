import numpy as np
import cv2
import matplotlib.pyplot as plt

img = cv2.imread("tap.png", 1)
a = 5
tx, ty = 30, 30
M = img.shape[0]
N = img.shape[1]
output = np.copy(img)

for i in range(img.shape[0]):
    for j in range(img.shape[1]):
        # print(np.sin((2 * np.pi * j) / tx))
        u = i + a * np.sin((2 * np.pi / tx) * (i - M))
        v = j + a * np.sin((2 * np.pi / ty) * (j - N))
        u = np.round(u).astype(np.uint32)
        v = np.round(v).astype(np.uint32)
        for k in range(3):
            if u < 515 and v < 807:
                output[i, j, k] = img[u, v, k]
            else:
                output[i, j, k] = img[i, j, k]

cv2.imshow("Original", img)
cv2.imshow("Output", output)
#cv2.imwrite("Output.jpg", output)
cv2.waitKey()
cv2.destroyAllWindows()
