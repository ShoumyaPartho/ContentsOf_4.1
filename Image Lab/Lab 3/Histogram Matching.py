# Solved By: Shoumya
# Roll: 1807021

import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import math


img = cv.imread("eye.png",cv.IMREAD_GRAYSCALE)
cv.imshow("Input Image",img)

plt.figure(figsize=(10,4))

plt.subplot(3,3,1)
plt.title("Input Image Histogram")
plt.hist(img.ravel(),256,[0,255])
plt.show()

freq = np.zeros(256,np.int32)

for i in range(img.shape[0]):
    for j in range(img.shape[1]):
        pix = int(img[i][j])
        freq[pix]+=1
        
pdf = np.zeros(256,np.float32)

for i in range(256):
    pdf[i] = freq[i]/(img.shape[0]*img.shape[1])
    
cdf = np.zeros(256,np.float32)
s = np.zeros(256,np.float32)

cdf[0] = pdf[0]

for i in range(1,256):
    cdf[i] = cdf[i-1]+pdf[i]
    
for i in range(256):
    s[i] = int(round(cdf[i]*255.0))

e = 2.718281
erlangPdf = np.zeros(256,np.float32)
erlangCdf = np.zeros(256,np.float32)
g = np.zeros(256,np.float32)
#k = 2
#miu = 30
print("Enter value of k: ")
k = int(input())

print("Enter value of miu: ")
miu = int(input())

for i in range(256):
    erlangPdf[i] = ((i**(k-1)) * (e**(-i/miu)))/((miu**k) * math.factorial(k-1))

erlangCdf[0] = erlangPdf[0]

for i in range(1,256):
    erlangCdf[i] = (erlangCdf[i-1]+erlangPdf[i])

for i in range(256):
    g[i] = int(round(erlangCdf[i] * 255))


# Searching for closest match in the lookup table of G(z)
for i in range(256):
    res = i
    mini = 10000
    
    for j in range(256):
        if abs(g[j]-s[i])<mini:
            mini = abs(g[j]-s[i])
            res = j
    s[i] = res

for i in range(img.shape[0]):
    for j in range(img.shape[1]):
        img[i][j] = s[img[i][j]]

cv.imshow("OUTPUT Image",img)

oFreq = np.zeros(256,np.int32)
oPdf = np.zeros(256,np.float32)
oCdf = np.zeros(256,np.float32)

for i in range(img.shape[0]):
    for j in range(img.shape[1]):
        pix = img[i][j]
        oFreq[pix] += 1

for i in range(256):
    oPdf[i] = oFreq[i]/(img.shape[0]*img.shape[1])

oCdf[0] = oPdf[0]

for i in range(256):
    oCdf[i] = oCdf[i-1] + oPdf[i]


    
plt.subplot(3,3,2)
plt.title("Input PDF")
plt.plot(pdf)
plt.show()

plt.subplot(3,3,3)
plt.title("Input CDF")
plt.plot(cdf)
plt.show()


plt.subplot(3,3,5)
plt.title("ERLANG PDF")
plt.plot(erlangPdf)
plt.show()

plt.subplot(3,3,6)
plt.title("ERLANG CDF")
plt.plot(erlangCdf)
plt.show()

plt.subplot(3,3,7)
plt.title("Output Image Histogram")
plt.hist(img.ravel(),256,[0,255])
plt.show()

plt.subplot(3,3,8)
plt.title("Output Image PDF")
plt.plot(oPdf)
plt.show()

plt.subplot(3,3,9)
plt.title("Output Image CDF")
plt.plot(oCdf)
plt.show()


cv.waitKey()
cv.destroyAllWindows()