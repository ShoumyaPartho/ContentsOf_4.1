import numpy as np
import cv2


def gaussian(m,n,sigma):
    gaussian = np.zeros((m,n))
    m = m//2
    n = n//2
    
    for x in range(-m,m+1):
        for y in range(-n,n+1):
            
            x1 = (2*np.pi * sigma**2)
            x2 = np.exp(-(x**2 + y**2)/(2*sigma**2))
            
            gaussian[x+m,y+n] = (1/x1) * x2
    
    return gaussian

def normalize(img):
    mx = img.max()
    mn = img.min()
    h,w = img.shape
    
    for i in range(h):
        for j in range(w):
            img[i,j] = int((img[i,j]-mn)/(mx-mn)*255)
            
    return img
    
def conv(img,filt):
    S = img.shape
    F = filt.shape

    R = S[0] + F[0] -1
    C = S[1] + F[1] -1

    Z = np.zeros((R,C))


    for x in range(S[0]):
        for y in range(S[1]):
            Z[x+int((F[0]-1)/2), y+int((F[1]-1)/2)] = img[x,y]


    for i in range(S[0]):
        for j in range(S[1]):
            k = Z[i:i+F[0],j:j+F[1]]
            sum = 0
            for x in range(F[0]):
                for y in range (F[1]):
                        sum += (k[x][y] * filt[-x][-y])

            img[i,j] = sum
    
    return img

def median(img,filt):
    S = img.shape
    F = filt.shape

    R = S[0] + F[0] -1
    C = S[1] + F[1] -1

    Z = np.zeros((R,C))


    for x in range(S[0]):
        for y in range(S[1]):
            Z[x+int((F[0]-1)/2), y+int((F[1]-1)/2)] = img[x,y]


    for i in range(S[0]):
        for j in range(S[1]):
            k = Z[i:i+F[0],j:j+F[1]]
            v = []
            for x in range(F[0]):
                for y in range (F[1]):
                    v.append(k[x][y])
            v.sort()
            img[i,j] = v[int(len(v)/2)]
            v.clear()
    return img

img = cv2.imread("Lena.jpg", cv2.IMREAD_GRAYSCALE)
cv2.imshow("Original",img) 

#plt.imshow(img,'gray')
#plt.show()


#Mean
filt = np.array([
        [1,1,1,1,1],
        [1,1,1,1,1],
        [1,1,1,1,1],
        [1,1,1,1,1],
        [1,1,1,1,1]])/25
cpy = img.copy()
im = conv(cpy, filt)
cv2.imshow("Mean",im)


#Median
filt = np.array([[0,0,0,0,0],
                 [0,0,0,0,0],
                 [0,0,0,0,0],
                 [0,0,0,0,0],
                 [0,0,0,0,0]])
cpy = img.copy()
im = median(cpy,filt)
cv2.imshow("Median",im)


#Gaussian
#sigma = int(input("Enter sigma value: "))
sigma = 3
m = 5*sigma
if m%2==0:
    m += 1
filt = gaussian(m,m,sigma)
cpy = img.copy()
im = conv(cpy, filt)
cv2.imshow("Gaussian",im)


#Laplace
filt = np.array([
        [0,1,0],
        [1,-4,1],
        [0,1,0]])
cpy = img.copy()
im = conv(cpy, filt)
cv2.imshow("Laplacian",im)

#Sobel
filt = np.array([
        [1,2,1],
        [0,0,0],
        [-1,-2,-1]])
cpy = img.copy()
im = conv(cpy, filt)
cv2.imshow("Sobel1",im)

filt = np.array([
        [1,0,-1],
        [2,0,-2],
        [1,0,-1]])
cpy = img.copy()
im = conv(cpy, filt)
cv2.imshow("Sobel2",im)
    

#print(filt)


        
#cv2.imshow("final",img)

#plt.imshow(img,'gray')
#plt.show()


cv2.waitKey(0)
cv2.destroyAllWindows()

