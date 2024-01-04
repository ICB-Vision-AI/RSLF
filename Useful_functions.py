from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import csv

#importations
def load_calib_param():
	dat = []
	with open('Data/calibration_param.csv', 'r') as f_:
		reader = csv.reader(f_)
		for row in reader:
			dat.append(np.float32(row[0]))
	F = dat[0]
	D = dat[1]
	O = np.ones((2,))
	O[0] = dat[2]
	O[1]= dat[3]
	f = dat[4]
	nb_s = dat[5]
	nb_t = dat[6]
	tau = dat[7]
	s = np.array(dat)[8:8+int(nb_s)]
	t = np.array(dat)[8+int(nb_s):]
	return F, D, O, f, s, t, tau

def load_rslf_image(folder):
	dat = []
	with open(folder+'/rslf_image.csv', 'r') as f_:
		reader = csv.reader(f_)
		for row in reader:
			dat.append(np.array(np.float32(row)))
	dat = np.array(dat)
	return dat

def set_axes_equal(ax):
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)

    # The plot bounding box is a sphere in the sense of the infinity
    # norm, hence I call half the max range the plot radius.
    plot_radius = 0.5*max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])
    
    ax.set_box_aspect([1,1,1])
	
