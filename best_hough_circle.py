#!/usr/bin/python3
import numpy as np
import cv2 as cv

def best_hough_circle(input_file, min_max_radii, best_guess_radius):
	# Arguments:
	# input file = file path to the input image
	# min_max_radii = tuple of format (min, max) indicating the minimum and 
	# 	maximum acceptable radii
	# best_guess_radius = best guess of the target circle radius, to help 
	# 	determine best circle in the case there are multiple
	# 
	# Returns:
	# [x,y,r] of best fit circle based on given parameters
	# x = x coordinate of center of circle
	# y = y coordinate of center of circle
	# y = radius of circle
	
	# separate min/max radius
	min_radius, max_radius = min_max_radii

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

	circles = np.uint16(np.around(circles))

	if circles is not None:
		circles = np.uint16(np.around(circles))[0,:]
		#print(circles)

		# filter out all circles that are outside the given min/max radii
		circles_filtered = \
			[x for x in circles if x[2] >= min_radius or x[2] <= max_radius]

		# if only one circle found, just return it
		if len(circles_filtered) == 1:
			return circles[0]
		else: # find best circle based on given best estimate
			match_idx = find_nearest(circles[:,2], best_guess_radius)
		return circles[match_idx]
	else:
		print("No best match found")
		return None

def find_nearest(array, value):
	array = np.asarray(array)
	idx = (np.abs(array - value)).argmin()
	return idx

if __name__ == "__main__":
	result = best_hough_circle('./images/gradient_circles.png', (0, 50), (25))
	print(result)