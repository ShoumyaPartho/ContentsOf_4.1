import cv2
import numpy as np

img = cv2.imread("input_img.jpg", 0)
r, img = cv2.threshold(img, 130, 255, cv2.THRESH_BINARY)  # _INV)
cv2.imshow("Original", img)
rate = 50
k1 = np.array([[0, 0, 0], [1, 1, 0], [1, 0, 0]], dtype=np.uint8)
k1 = cv2.resize(k1, None, fx=rate, fy=rate, interpolation=cv2.INTER_NEAREST)

k2 = np.array([[0, 1, 1], [0, 0, 1], [0, 0, 1]], dtype=np.uint8)
k2 = cv2.resize(k2, None, fx=rate, fy=rate, interpolation=cv2.INTER_NEAREST)
k3 = np.array([[1, 1, 1], [0, 1, 0], [0, 1, 0]], dtype=np.uint8)
k3 = cv2.resize(k3, None, fx=rate, fy=rate, interpolation=cv2.INTER_NEAREST)

kernel = np.ones((150, 150), np.uint8)
d = np.ones((20, 20), np.uint8)

b1 = np.copy(k1)
b2 = kernel - k1
output1 = cv2.erode(img, b1, iterations=1)
tmp1 = 255 - img
tmp1 = cv2.erode(tmp1, b2, iterations=1)
output1 = cv2.bitwise_and(output1, tmp1)
output1 = cv2.dilate(output1, d, iterations=1)
cv2.imshow("Output1", output1)

bb1 = np.copy(k2)
bb2 = kernel - k2
output2 = cv2.erode(img, bb1, iterations=1)
tmp2 = 255 - img
tmp2 = cv2.erode(tmp2, bb2, iterations=1)
output2 = cv2.bitwise_and(output2, tmp2)
output2 = cv2.dilate(output2, d, iterations=1)
cv2.imshow("Output2", output2)

bbb1 = np.copy(k3)
bbb2 = kernel - k3
output3 = cv2.erode(img, bbb1, iterations=1)
tmp3 = 255 - img
tmp3 = cv2.erode(tmp3, bbb2, iterations=1)
output3 = cv2.bitwise_and(output3, tmp3)
output3 = cv2.dilate(output3, d, iterations=1)
cv2.imshow("Output3", output3)

k1 = k1 * 255
cv2.imshow("k1", k1)
k2 = k2 * 255
cv2.imshow("k2", k2)
k3 = k3 * 255
cv2.imshow("k3", k3)

cv2.waitKey(0)
cv2.destroyAllWindows()
