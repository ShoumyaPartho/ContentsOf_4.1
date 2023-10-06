import numpy as np
import cv2
import matplotlib.pyplot as plt

img = cv2.imread("flower.jpg", 1)

print(img.shape)
ax = 10
ay = 10
tx, ty = 20, 20
# ax = 10
# ay = 15
# tx, ty = 50, 70

output = np.copy(img)
for i in range(img.shape[0]):
    for j in range(img.shape[1]):
        # print(np.sin((2 * np.pi * j) / tx))
        u = i + ax * np.sin((2 * np.pi * j) / tx)
        v = j + ay * np.sin((2 * np.pi * i) / ty)
        u = np.round(u).astype(np.uint32)
        v = np.round(v).astype(np.uint32)
        for k in range(3):
            if 0 <= u < 407 and 0 <= v < 611:
                output[i, j, k] = img[u, v, k]
            else:
                output[i, j, k] = 0

cv2.imshow("Original", img)
# cv2.imwrite("Output", output)
cv2.imshow("Output", output)
cv2.waitKey()
cv2.destroyAllWindows()
