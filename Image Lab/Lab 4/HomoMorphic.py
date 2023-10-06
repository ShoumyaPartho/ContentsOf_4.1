# -*- coding: utf-8 -*-
"""
Created on Mon May 27 11:44:06 2023

@author: SHOUMYA
"""

import cv2
import math
import numpy as np
import matplotlib.pyplot as plt

def apply_illuminating_effect(image, angle):
    # Apply illuminating effect
    rows, cols = image.shape
    #center = (cols // 2, rows // 2)

    # Create a meshgrid for pixel coordinates
    #x, y = np.meshgrid(np.arange(cols), np.arange(rows))
    
    # Convert angle to radians
    #theta = np.radians(angle)
    
    # Calculate the distance from the center
    #radius = np.sqrt((x - center[0])**2 + (y - center[1])**2)
    
    # Calculate the angle of each pixel
    #phi = np.arctan2(y - center[1], x - center[0])
    
    # Calculate the amount of illumination
    #illumination = 1.0 - np.cos(2.0 * np.pi * radius / cols + theta)
    
    width, height = cols, rows
    angle = np.deg2rad(angle)

    x_ara = np.linspace(-1,1,width)
    y_ara = np.linspace(-1,1,height)
    x_mara, y_mara = np.meshgrid(x_ara, y_ara)

    grad_dir = np.array([np.cos(angle), np.sin(angle)])
    illumination = grad_dir[0] * x_mara + grad_dir[1] * y_mara

    illumination -= illumination.min()
    illumination /= illumination.max()
    
    #cv2.imshow("Illumination", illumination)
    
    plt.subplot(2, 4, 1)
    plt.imshow(illumination, cmap='gray')
    plt.title('Illumination')
    #cv2.imwrite('H:/Academic/4-1/Image Lab/Lab 4/Ass/illumination.jpg',illumination)

    # Apply the illuminating effect
    result = image * illumination

    return result.astype(np.uint8)

# Load the grayscale image
image = cv2.imread('homo.jpg', cv2.IMREAD_GRAYSCALE)

gh = 1.2
gl = 0.5
c = 0.1
d0 = 50
angle = 45

'''angle = float(input("Angle: "))
gl = float(input("Gamma Low: "))
gh = float(input("Gamma High: "))
c = float(input("C: "))
d0 = float(input("D0: ")) '''

# Apply illuminating effect at an angle of 45 degrees
illuminating_effect = apply_illuminating_effect(image, angle)

# Display the original and transformed images
plt.subplot(2, 4, 2)
plt.imshow(image, cmap='gray')
plt.title('Original Image')
#cv2.imshow("Original Image", image)

plt.subplot(2, 4, 3)
plt.imshow(illuminating_effect, cmap='gray')
plt.title('Corrupted Image')
#cv2.imshow("Corrupted Image", illuminating_effect)

img = np.log1p(illuminating_effect)

filt = np.zeros((img.shape[0],img.shape[1]), np.float32)




for i in range(img.shape[0]):
    for j in range(img.shape[1]):
        u = (i-img.shape[0]/2)**2
        v = (j-img.shape[1]/2)**2
        r = math.exp(-c*((u+v)/d0**2))
        r = (gh-gl)*(1-r)+gl
        filt[i][j] = r

plt.subplot(2,4,4)
plt.imshow(filt,'gray')
plt.title("Homomorphic filter:")
#cv2.imshow("Homomorphic Filter", filt)

f = np.fft.fft2(img)

shift = np.fft.fftshift(f)

mag = np.abs(shift)

plt.subplot(2,4,5)
plt.imshow(np.log(mag),'gray')
plt.title("Magnitude before multiplication with filter:")
#cv2.imshow("Magnitude before multiplication with filter", np.log(mag))

phase = np.angle(shift)

mag = mag*filt

plt.subplot(2,4,6)
plt.imshow(np.log(mag),'gray')
plt.title("Magnitude after multiplication with filter:")
#cv2.imshow("Magnitude after multiplication with filter", np.log(mag))


op = np.multiply(mag,np.exp(1j*phase))

ishift = np.fft.ifftshift(op)

inv = np.real(np.fft.ifft2(ishift))

inv = np.expm1(inv)

plt.subplot(2,4,7)
plt.imshow(inv,'gray')
plt.title("Output image:")
#cv2.imshow("Inverse", inv)

plt.show()