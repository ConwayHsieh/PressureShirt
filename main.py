#!/usr/bin/python3
import os, json
import scipy.optimize as opt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from playsound import playsound
from scipy.spatial import distance

def main():
	# load configs
	with open("config.json", "r") as f:
		config = json.load(f)
	vol = config["control"]["volume"]
	sound_dir = config["SoundFiles"]["dir"]
	sound_files_dict = config["SoundFiles"]["files_dict"]
	#print(sound_files_dict)
	#print(len(sound_files_dict))

	grid_rows = config["attributes"]["rows"]
	grid_cols = config["attributes"]["columns"]

	if grid_rows*grid_cols != len(sound_files_dict):
		raise Exception(\
			"Input grid and corresponding sound files dictionary do not match in size")

	# generating test data
	# TODO interface with data intake
	'''
	xy, z = gen_test_data()
	#print(xy)
	'''
	xy, z = gen_rng_data(25, 0.54, 0.73)
	#xy = np.transpose(xy)
	print(xy)
	x = xy[0,:]
	y = xy[1,:]
	#print(x)
	#print(y)
	#print(z)

	i = z.argmax()
	guess = [1, x[i], y[i], 1, 1, 1]
	pred_params, uncert_cov = opt.curve_fit(gauss2d, xy, z, p0=guess)
	#print(pred_params)
	pred_xy = pred_params[1:3]
	print('Predicted xy: ')
	print(pred_xy)

	grid = gen_grid(rows=grid_rows, cols=grid_cols)
	grid_x, grid_y = grid
	#print(grid)
	#print(grid.shape)

	ax = plot(xy, z, pred_params)
	ax.scatter(grid_x, grid_y, marker='*')

	closest_val, closest_val_index = find_closest_point(pred_xy, grid)
	print(closest_val)
	print(closest_val_index)

	ax.plot(closest_val[0], closest_val[1], marker='x')

	plt.show()

	playsound(os.path.join(sound_dir,sound_files_dict[str(closest_val_index)]))

	return

def gauss2d(xy, amp, x0, y0, a, b, c):
	x = xy[0,:]
	y = xy[1,:]
	inner = a * (x - x0)**2 
	inner += 2 * b * (x - x0)**2 * (y - y0)**2
	inner += c * (y - y0)**2
	return amp * np.exp(-inner)

def plot(xy, zobs, pred_params):
	x = xy[0,:]
	y = xy[1,:]
	xmax = np.amax(x)
	ymax = np.amax(y)

	pred_xy = pred_params[1:3]

	yi, xi = np.mgrid[:xmax:100j, :ymax:100j]
	xyi = np.vstack([xi.ravel(), yi.ravel()])

	zpred = gauss2d(xyi, *pred_params)
	zpred.shape = xi.shape

	fig, ax = plt.subplots()
	ax.scatter(x, y, c=zobs, s=500, linewidths=5,
		vmin=zpred.min(), vmax=zpred.max())
	ax.plot(pred_xy[0], pred_xy[1], marker = 'o', markerfacecolor="red")
	im = ax.imshow(zpred, extent=[xi.min(), xi.max(), yi.max(), yi.min()],
				   aspect='auto')
	fig.colorbar(im)
	ax.invert_yaxis()
	return ax

def gen_test_data():
	data = np.array([
				[0,0,0.5],
				[0,1,0.5],
				[0,2,0.5],
				[1,0,0.5],
				[1,1,1],
				[1,2,0.5],
				[2,0,0.5],
				[2,1,0.5],
				[2,2,0.5]])

	return data[:,0:2], data[:,2]

def gen_rng_data(num, x0, y0):
    xy = np.random.random((2,num))

    params = [1, x0, y0, 1,2,3]
    zobs = gauss2d(xy, *params)
    return xy, zobs

def gen_grid(rows=2, cols=2):
	rows = rows+1
	cols = cols+1
	max_val = 0.99999
	return np.mgrid[1/cols:max_val:1/cols,
					1/rows:max_val:1/rows]

def find_closest_point(xy, grid):
	grid = grid.reshape(2,-1).transpose()
	print(grid)
	dist = np.array([distance.euclidean(g, xy) for g in grid])
	min_val_index = np.argmin(dist)
	return grid[min_val_index], min_val_index



if __name__ == "__main__":
	main()