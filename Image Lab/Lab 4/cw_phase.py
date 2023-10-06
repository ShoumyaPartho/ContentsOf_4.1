# Fourier transform - guassian lowpass filter

import cv2
import numpy as np
from matplotlib import pyplot as plt
from copy import deepcopy as dpc



def min_max_normalize(img_inp):
    inp_min = np.min(img_inp)
    inp_max = np.max(img_inp)

    for i in range (img_inp.shape[0]):
        for j in range(img_inp.shape[1]):
            img_inp[i][j] = (((img_inp[i][j]-inp_min)/(inp_max-inp_min))*255)
    return np.array(img_inp, dtype='uint8')

# take input
img_input = cv2.imread('period_input4.jpeg', 0)

img = dpc(img_input)

image_size = img.shape[0] * img.shape[1]

# fourier transform
ft = np.fft.fft2(img)

ft_shift = np.fft.fftshift(ft)



magnitude_spectrum_ac=np.abs(ft_shift)
magnitude_spectrum = 1 * np.log(np.abs(ft_shift)+1)
magnitude_spectrum_scaled = min_max_normalize(magnitude_spectrum)
cv2.imshow("Magnitude Spectram Before Multiplying Filter", magnitude_spectrum_scaled)
cv2.imwrite('H:/Academic/4-1/Image Lab/Lab 4/Ass/magnitude_spectrum_scaled.jpg',magnitude_spectrum_scaled)

notch_filt = np.zeros((img.shape[0],img.shape[1]),np.int32)
notch_filt2 = np.zeros((img.shape[0],img.shape[1]),np.int32)

print("Enter coordinate x: ")
x = int(input())
xn = img.shape[0]-x
print("Enter coordinate y: ")
y = int(input())
yn = img.shape[1]-y
print("Enter radius r: ")
r = int(input())

for i in range(img.shape[0]):
    for j in range(img.shape[1]):
        dis = (((i-x)**2)+((j-y)**2))**0.5
        dis2 = (((i-xn)**2)+((j-yn)**2))**0.5
        if(dis<=r):
            notch_filt[i][j] = 0
        else:
            notch_filt[i][j] = 1
            
        if(dis2<=r):
            notch_filt2[i][j] = 0
        else:
            notch_filt2[i][j] = 1

'''for i in range(img.shape[0]):
    for j in range(img.shape[1]):
        dis = (((i-xn)**2)+((j-yn)**2))**0.5
        if(dis<=r):
            notch_filt2[i][j] = 0
        else:
            notch_filt2[i][j] = 1'''

magnitude_spectrum_ac = magnitude_spectrum_ac*notch_filt*notch_filt2



#mag, ang = cv2.cartToPolar(ft_shift[:,:,0],ft_shift[:,:,1])
ang = np.angle(ft_shift)

## phase add
final_result = np.multiply(magnitude_spectrum_ac, np.exp(1j*ang))
#final_result2 = np.multiply(magnitude_spectrum, np.exp(1j*ang))

# inverse fourier
img_back = np.real(np.fft.ifft2(np.fft.ifftshift(final_result)))
#img_back2 = np.real(np.fft.ifft2(np.fft.ifftshift(final_result2)))
img_back_scaled = min_max_normalize(img_back)
#img_back_scaled2 = min_max_normalize(img_back2)

## plot
cv2.imshow("input", img_input)
cv2.imwrite('H:/Academic/4-1/Image Lab/Lab 4/Ass/img_input.jpg',img_input)
cv2.imshow("Magnitude Spectrum",min_max_normalize(magnitude_spectrum_scaled*notch_filt*notch_filt2))
cv2.imwrite('H:/Academic/4-1/Image Lab/Lab 4/Ass/Magnitude Spectrum.jpg',min_max_normalize(magnitude_spectrum_scaled*notch_filt*notch_filt2))
#cv2.imshow("Magnitude Spectrum2",magnitude_spectrum_scaled2)
#print(notch_filt | notch_filt2)
#temp=notch_filt & notch_filt2

#min_max_normalize(temp)
#print(temp.type)
cv2.imshow("Phase",ang)
cv2.imwrite('H:/Academic/4-1/Image Lab/Lab 4/Ass/Phase.jpg',ang)
cv2.imshow("Inverse transform",img_back_scaled)
cv2.imwrite('H:/Academic/4-1/Image Lab/Lab 4/Ass/Inverse transform.jpg',img_back_scaled)
#cv2.imshow("Notch_Filter",temp )
#cv2.imshow("Phase2",ang)
#cv2.imshow("Inverse transform",img_back_scaled2)



cv2.waitKey(0)
cv2.destroyAllWindows() 
