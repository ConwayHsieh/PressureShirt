#!/usr/bin/python3
import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

# import image
img = cv.imread('./images/gradient_circles.png')
assert img is not None, "file could not be read, check with os.path.exists()"

# binarization to approximate locations of interest // thresholding
gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
ret, thresh = cv.threshold(gray,0,255,cv.THRESH_BINARY_INV+cv.THRESH_OTSU)

# noise removal using morphological opening
# closing small holes in objects using morphological closing
kernel = np.ones((3,3),np.uint8)
cleaned_openings = cv.morphologyEx(thresh,cv.MORPH_OPEN,kernel, iterations = 2)
cleaned = cv.morphologyEx(cleaned_openings,cv.MORPH_CLOSE,kernel, iterations = 2)

# Using Hough Transform to detect circles
rows = gray.shape[0]
circles = cv.HoughCircles(gray,cv.HOUGH_GRADIENT,1,rows/8,
                            param1=50,param2=30,minRadius=0,maxRadius=0)

# draw circles on img
if circles is not None:
    circles = np.uint16(np.around(circles))[0,:]
    print(len(circles))
    print(circles)
    print(circles[:,2])
    for i in circles:
        center = (i[0], i[1])
        # circle center
        cv.circle(img, center, 1, (0, 100, 100), 3)
        # circle outline
        radius = i[2]
        cv.circle(img, center, radius, (255, 0, 255), 3)

cv.imshow('detected circles',img)
cv.waitKey(0)
cv.destroyAllWindows()
