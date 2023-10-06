import numpy as np
import cv2
import math as mt

path = "lena.jpg"

img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
cv2.imshow("img", img)
img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)  # cv2.imread(path, 0)
kernel = np.array(([0, -1, 0], [-1, 5, -5], [0, -1, 2]), np.float32)
x = mt.sqrt(kernel.size)
kernel_size = int(x)
pad = kernel_size // 2
pad = int(pad)
borderedImage = cv2.copyMakeBorder(img, pad, pad, pad, pad, cv2.BORDER_REPLICATE)
im_H = borderedImage.shape[0]
im_W = borderedImage.shape[1]
result4 = np.zeros((512, 512), dtype="float32")
for x in range(1, 513):
    for y in range(1, 513):
        for i in range(3):
            for j in range(3):
                result4[x - 1][y - 1] += (
                    borderedImage[x + i - 1][y + j - 1] * kernel[2 - i][2 - j]
                )
cv2.imshow("Output", result4)
cv2.imwrite("Output.jpg", result4)
cv2.waitKey(0)
cv2.destroyAllWindows()
