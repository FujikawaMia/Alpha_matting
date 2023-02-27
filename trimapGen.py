import cv2 as cv
import numpy as np

image_path  = str(input('give a path of your alpha image (png format required):'))
mask = cv.imread(image_path, 0)

kernel_dilate = np.ones((3, 3))
dilate = cv.dilate(mask,kernel_dilate, iterations=6)

kernel_erode = np.ones((3, 3))
erode  = cv.erode(mask,kernel_erode, iterations=10)

mask[mask==1] = 255
gray = cv.bitwise_xor(erode,dilate)
gray[gray!=0]=105

for i in range(erode.shape[0]):
    for j in range(erode.shape[1]):
        if(erode[i][j] != 0):
            gray[i][j] = 255

cv.imwrite('trimap.png',gray)
cv.imshow('trimp',gray)
cv.waitKey(0)