import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt


img = cv.imread("eye.png",cv.IMREAD_GRAYSCALE)
cv.imshow("input",img)

histr = cv.calcHist([img],[0],None,[256],[0,255])
#plt.plot(histr)

plt.figure(figsize=(10,4))

plt.subplot(1,4,1)
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

cdf[0] = pdf[0]

for i in range(1,256):
    cdf[i] = cdf[i-1]+pdf[i]

plt.subplot(1,4,2)
plt.title("Input Image CDF")
plt.plot(cdf)
plt.show()
    
for i in range(256):
    cdf[i] = round(cdf[i]*255.0)

    
for i in range(img.shape[0]):
    for j in range(img.shape[1]):
        #pix = int(round(img[i][j]))
        img[i][j] = cdf[img[i][j]]
        
plt.subplot(1,4,3)
plt.title("Output Image Histogram")
plt.hist(img.ravel(),256,[0,255])
plt.show()

cv.imshow("OUTPUT",img)

newFreq = np.zeros(256,np.int32)
newPdf = np.zeros(256,np.float32)
newCdf = np.zeros(256,np.float32)

for i in range(img.shape[0]):
    for j in range(img.shape[1]):
        pix = int(img[i][j])
        newFreq[pix] += 1

for i in range(256):
    newPdf[i] = newFreq[i]/(img.shape[0]*img.shape[1])

newCdf[0] = newPdf[0]

for i in range(1,256):
    newCdf[i] = newCdf[i-1]+newPdf[i]

plt.subplot(1,4,4)
plt.title("Output Image CDF")
plt.plot(newCdf)
plt.show()

cv.waitKey()
cv.destroyAllWindows()


cv.waitKey()
cv.destroyAllWindows()