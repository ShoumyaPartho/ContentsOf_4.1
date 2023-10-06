# -*- coding: utf-8 -*-
"""
Created on Tue May 16 11:45:54 2023

@author: USER
"""
import cv2
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from copy import deepcopy as dpc

matplotlib.use("TkAgg")

point_list = []


def min_max_normalize(img_inp):
    inp_min = np.min(img_inp)
    inp_max = np.max(img_inp)

    for i in range(img_inp.shape[0]):
        for j in range(img_inp.shape[1]):
            img_inp[i][j] = ((img_inp[i][j] - inp_min) / (inp_max - inp_min)) * 255
    return np.array(img_inp, dtype="uint8")


img_input = cv2.imread("two_noise.jpeg", 0)
img = dpc(img_input)
image_size = img.shape[0] * img.shape[1]

# fourier transform
ft = np.fft.fft2(img)

ft_shift = np.fft.fftshift(ft)

magnitude_spectrum_ac = np.abs(ft_shift)
magnitude_spectrum = 20 * np.log(np.abs(ft_shift))
magnitude_spectrum_scaled = min_max_normalize(magnitude_spectrum)

mag_img = dpc(magnitude_spectrum_scaled)

# img = np.zeros((10,12), np.uint8)
# img[4:6, 5:7] = 1
# click and seed point set up
x = None
y = None


# The mouse coordinate system and the Matplotlib coordinate system are different, handle that
def onclick(event):
    global x, y
    ax = event.inaxes
    if ax is not None:
        x, y = ax.transData.inverted().transform([event.x, event.y])
        x = int(round(x))
        y = int(round(y))
        print(
            "button=%d, x=%d, y=%d, xdata=%f, ydata=%f"
            % (event.button, event.x, event.y, x, y)
        )
        point_list.append((x, y))


X = np.zeros_like(mag_img)
plt.title("Please select seed pixel from the input")
im = plt.imshow(mag_img, cmap="gray")
im.figure.canvas.mpl_connect("button_press_event", onclick)
plt.show(block=True)


n = 2
d = 30
M = mag_img.shape[0]
N = mag_img.shape[1]

# point_list.append((M // 2, N // 2))
print(point_list)
operation = np.zeros_like(mag_img, dtype=np.float32)


for i in range(mag_img.shape[0]):
    for j in range(mag_img.shape[1]):
        Hr = 1
        for k in point_list:
            dk = 1
            dk = (k[0] - M // 2 - i) ** 2 + (k[1] - N // 2 - j) ** 2
            dk = np.sqrt(dk)

            dkk = (k[0] - M // 2 + i) ** 2 + (k[1] - N // 2 + j) ** 2
            dkk = np.sqrt(dkk)
            if dk != 0 and dkk != 0:
                a = (1 / (1 + (d / dk)** (2 * n))) * (1 / (1 + (d / dkk)** (2 * n)) )
            Hr = Hr * a
        operation[i, j] = Hr
# min_max_normalize(operation)
# print(operation)
# operation = cv2.normalize(operation, None, 0, 255, cv2.NORM_MINMAX)

ang = np.angle(ft_shift)

## phase add
final_result = np.multiply(magnitude_spectrum_ac * operation, np.exp(1j * ang))

# inverse fourier
img_back = np.real(np.fft.ifft2(np.fft.ifftshift(final_result)))
img_back_scaled = min_max_normalize(img_back)

## plot
cv2.imshow("input", img_input)
cv2.imshow("Magnitude Spectrum", magnitude_spectrum_scaled)

cv2.imshow("Inverse transform", img_back_scaled)


cv2.waitKey(0)
cv2.destroyAllWindows()
