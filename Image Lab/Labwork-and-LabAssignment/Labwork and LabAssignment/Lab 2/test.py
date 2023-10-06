import numpy as np
import cv2
import math


def spatial_gaussian_kernel(ksize, sigma):
    kernel = np.zeros((ksize, ksize), dtype="float32")
    k = ksize // 2
    gconst = sigma**2
    gconst = 2 * math.pi * gconst
    for i in range(-k, k + 1):
        for j in range(-k, k + 1):
            kernel[i + k][j + k] = (
                math.exp(-((i * i) + (j * j)) / (2 * sigma**2))
            ) / gconst
    return kernel


def range_kernel(img, x, y, s):
    k = []
    s = s // 2
    for i in range(-s, s):
        for j in range(-s, s):
            k.append(img[x + i][y + j])
    k.sort()
    return k


img = cv2.imread("lena.png", cv2.IMREAD_GRAYSCALE)
ksize = 5
sigma = 5
height = img.shape[0]
width = img.shape[1]
output = np.zeros((height, width), dtype="float32")
k = ksize // 2
bordered_output = cv2.copyMakeBorder(img, k, k, k, k, cv2.BORDER_REPLICATE)
output = np.zeros_like(img)

sp_kernel = spatial_gaussian_kernel(ksize, 4)
for i in range(bordered_output.shape[0]):
    for j in range(bordered_output.shape[1]):
        rng_kernel = range_kernel(img, i + k, j + k, ksize)
        final_kernel = rng_kernel * sp_kernel
        final_kernel = final_kernel / final_kernel.sum()
        for x in range(ksize):
            for y in range(ksize):
                output[i][j] += (
                    bordered_output[i + x][j + y]
                    * final_kernel[ksize - 1 - x][ksize - 1 - y]
                )

cv2.normalize(output, output, 0, 255, cv2.NORM_MINMAX)
output = np.round(output).astype(np.uint8)
cv2.imshow("Actual", img)
cv2.imshow("Biliteral", output)
cv2.waitKey(0)
cv2.destroyAllWindows()
