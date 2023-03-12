import cv2 as cv
import numpy as np

def trimapGenerator():
    alpha_path  = str(input('give a path (png format required):'))
    trimap_path = alpha_path[:len(alpha_path)-9] + '_trimap.png'
    mask = cv.imread(alpha_path, 0)
    mask = cv.resize(mask, (480,640))

    kernel_dilate = np.ones((3, 3))
    dilate = cv.dilate(mask,kernel_dilate, iterations=3)

    kernel_erode = np.ones((3, 3))
    erode  = cv.erode(mask,kernel_erode, iterations=5)

    mask[mask==1] = 255
    gray = cv.bitwise_xor(erode,dilate)
    gray[gray!=0]=105

    for i in range(erode.shape[0]):
        for j in range(erode.shape[1]):
            if(erode[i][j] != 0):
                gray[i][j] = 255

    cv.imwrite(trimap_path, gray)

if __name__ == "__main__":
    trimapGenerator()