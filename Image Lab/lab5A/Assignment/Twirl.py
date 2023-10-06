# -*- coding: utf-8 -*-
"""
Created on Tue Jun 13 05:42:59 2023

@author: SHOUMYA
"""

import cv2
import numpy as np

def twirl_effect(cpy, cx, cy, alpha,rm):
    h, w = cpy.shape
    output = cpy.copy()
    
    for x in range(w):
        for y in range(h):
            dx = x - cx
            dy = y - cy
            r = np.sqrt(dx**2 + dy**2)
            beta = np.arctan2(dy, dx) + alpha * ((rm-r)/rm)
            nx = int(cx + r * np.cos(beta))
            ny = int(cy + r * np.sin(beta))

            if 0 <= nx < w and 0 <= ny < h and r<=rm:
                output[y, x] = cpy[ny, nx]

    return output

def twirl_map(cpy,x, y, cx, cy, alpha,rm):
    dx = x - cx
    dy = y - cy
    r = np.sqrt(dx**2 + dy**2)
    beta = np.arctan2(dy, dx) + alpha * ((rm-r)/rm)
    nx = cx + r * np.cos(beta)
    ny = cy + r * np.sin(beta)
    
    check = r > rm
    
    nx[check] = dx[check] + cx
    ny[check] = dy[check] + cy
    
    remapped_img = cv2.remap(cpy, nx.astype(np.float32), ny.astype(np.float32), cv2.INTER_LINEAR)
    
    return remapped_img
    
img = cv2.imread("pic.jpg", cv2.IMREAD_GRAYSCALE)
cpy = img.copy()
h, w = img.shape

'''cx = w // 2  # Center X coordinate
cy = h // 2  # Center Y coordinate
alpha = 90  # Twirl angle'''

cx = int(input("Enter Xc: "))   # Center X coordinate
cy = int(input("Enter Yc: "))   # Center Y coordinate
alpha = int(input("Enter angle in degree: "))    # Twirl angle

rm = min(cx,cy) #Rmax

twirl_img = twirl_effect(cpy, cx, cy, np.radians(alpha),rm)

#x, y = np.meshgrid(np.arange(w), np.arange(h))
#print("x is : ", x)
#print("y is : ",y)
#twirl_img = twirl_map(cpy,x,y, cx, cy, np.radians(alpha),rm)

cv2.imshow("Input", img)
cv2.imshow("Twirl", twirl_img)

cv2.waitKey(0)
cv2.destroyAllWindows()

