#!/usr/bin/python3
import argparse
import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

def thresh_callback(val):
	threshold = val
	
	# binarization to approximate locations of interest // thresholding
	gray = cv.cvtColor(src,cv.COLOR_BGR2GRAY)
	ret, thresh = cv.threshold(gray,0,255,cv.THRESH_BINARY_INV+cv.THRESH_OTSU)

	# noise removal using morphological opening
	# closing small holes in objects using morphological closing
	kernel = np.ones((3,3),np.uint8)
	cleaned_openings = cv.morphologyEx(src,cv.MORPH_OPEN,kernel, iterations = 2)
	cleaned = cv.morphologyEx(cleaned_openings,cv.MORPH_CLOSE,kernel, iterations = 2)

	canny_output = cv.Canny(cleaned, threshold, threshold * 2)
	contours, _ = cv.findContours(canny_output, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
	
	drawing = np.zeros((canny_output.shape[0], canny_output.shape[1], 3), dtype=np.uint8)

	minEllipse = [None]*len(contours)
	color=(0,100,100)
	for i, c in enumerate(contours):
		if c.shape[0] > 5:
			minEllipse[i] = cv.fitEllipse(c)
			cv.ellipse(drawing, minEllipse[i], color, 2)

	cv.imshow('Contours', drawing)

# load parser and image
parser = argparse.ArgumentParser(description='Ellipses Contours')
parser.add_argument('--input', help='Path to input image.', default='./images/gradient_circles.png')
args = parser.parse_args()
src = cv.imread(cv.samples.findFile(args.input))
if src is None:
	print('Could not open or find the image:', args.input)
	exit(0)

source_window = 'Source'
cv.namedWindow(source_window)
cv.imshow(source_window, src)

max_thresh = 255
thresh = 100 # initial threshold
cv.createTrackbar('Canny Thresh:', source_window, thresh, max_thresh, thresh_callback)
thresh_callback(thresh)
cv.waitKey()