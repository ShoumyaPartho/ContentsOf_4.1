import numpy as np
import cv2
import math


def spatial_kernel(ksize):
    k = ksize // 2
    d = ksize // 2
    kernel = np.zeros((ksize, ksize), dtype=np.float32)
    for i in range(-k, k + 1):
        for j in range(-k, k + 1):
            r = np.sqrt(i**2 + j**2)
            kernel[i + k][j + k] = 1 - (r / d) ** 2
            if kernel[i + k][j + k] < 0:
                kernel[i + k][j + k] = 0
    print(kernel)
    return kernel


def range_kernel(img, x, y, ksize, sigma):
    kernel = np.zeros((ksize, ksize), dtype=np.float64)
    gconst = np.sqrt(2 * math.pi) * sigma
    k = ksize // 2
    Ip = img[x][y]
    for i in range(-k, k + 1):
        for j in range(-k, k + 1):
            Iq = img[x + i][y + j]
            kernel[i + k][j + k] = (
                math.exp(-((Ip - Iq) ** 2) / (2 * sigma * sigma))
            ) / gconst
    return kernel


img = cv2.imread("cube.png", 0)
ksize = 5
sigma = 80

sp_kernel = spatial_kernel(ksize)
k = ksize // 2
output = np.zeros_like(img, dtype=np.float32)
b_output = cv2.copyMakeBorder(img, k, k, k, k, cv2.BORDER_REPLICATE)


for i in range(img.shape[0]):
    for j in range(img.shape[1]):
        rng_kernel = range_kernel(b_output, i + k, j + k, ksize, sigma)
        final_kernel = sp_kernel * rng_kernel
        final_kernel = final_kernel / final_kernel.sum()
        for x in range(ksize):
            for y in range(ksize):
                output[i][j] += (
                    b_output[i + x][j + y] * final_kernel[ksize - 1 - x][ksize - 1 - y]
                )

cv2.normalize(output, output, 0, 255, cv2.NORM_MINMAX)
output = np.round(output).astype(np.uint8)

cv2.imshow("Actual", img)
cv2.imshow("Output", output)
cv2.waitKey(0)
cv2.destroyAllWindows()
