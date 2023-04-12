#!/usr/bin/python3
import json
import scipy.optimize as opt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def main():
	with open("config.json", "r") as f:
		config = json.load(f)

	#vol = config["control"]["volume"]
	
	xy, z = gen_test_data()
	xy = np.transpose(xy)
	#print(xy)
	x = xy[0,:]
	y = xy[1,:]
	print(x)
	print(y)
	print(z)

	i = z.argmax()
	guess = [1, x[i], y[i], 1, 1, 1]
	pred_params, uncert_cov = opt.curve_fit(gauss2d, xy, z, p0=guess)
	print(pred_params)
	print(pred_params[1:3])
	print("BREAK")

	print('done')

	plot(xy, z, pred_params)
	plt.show()

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

	yi, xi = np.mgrid[:xmax:100j, :ymax:100j]
	xyi = np.vstack([xi.ravel(), yi.ravel()])

	zpred = gauss2d(xyi, *pred_params)
	zpred.shape = xi.shape

	fig, ax = plt.subplots()
	ax.scatter(x, y, c=zobs, s=500, linewidths=5,
		vmin=zpred.min(), vmax=zpred.max())
	im = ax.imshow(zpred, extent=[xi.min(), xi.max(), yi.max(), yi.min()],
				   aspect='auto')
	fig.colorbar(im)
	ax.invert_yaxis()
	return fig

def gen_test_data():
	'''
	data = np.array([[0,0,0.5],
					[0,1,0.5],
					[0,2,0.5],
					[1,0,0.5],
					[1,1,1],
					[1,2,0.5],
					[2,0,0.5],
					[2,1,0.5],
					[2,2,0.5]])
	'''
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


if __name__ == "__main__":
	main()