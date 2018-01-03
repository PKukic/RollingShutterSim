""" Simulation of a meteor captured by a rolling shutter camera.
"""


# Python 2/3 compatibility
from __future__ import print_function, division, absolute_import
import numpy as np

import SimulationTools as st
import Parameters as par

# Customised parameters
rolling_shutter = True

show_plots = False

### Difference as a function of angular velocity and Y centroid coordinate ###

omega_ycentr_diff_arr = []

for omega_iter in par.omega_oyd_arr:

	# Get meteor duration
	t_meteor = st.timeFromAngle(par.phi, omega_iter, par.img_x, par.img_y, par.scale, par.fps)

	print("Meteor duration: {:.2f}".format(t_meteor))

	# LISTS of centroid and model coordinates
	centroid_coordinates, model_coordinates = st.pointsCentroidAndModel(rolling_shutter, t_meteor, par.phi, \
        omega_iter, par.img_x, par.img_y, par.scale, par.fps, par.sigma_x, par.sigma_y, par.noise_scale, par.offset, show_plots)

	if centroid_coordinates != -1 and model_coordinates != -1:

		# Size of frame number array
		frame_num_range = len(centroid_coordinates)
		print("Number of frames: {}".format(frame_num_range))

		for frame_num in range(frame_num_range):

			# Model-centroid point difference
			diff = st.centroidDifference(centroid_coordinates[frame_num], model_coordinates[frame_num])

			# Centroid coordinates
			y_centr = centroid_coordinates[frame_num][1]

			# Angular velocity in [px/s]
			omega_pxs = omega_iter * par.scale

			# Checking parameters
			print("Ang. velocity: {:.2f}; Y centroid coordinate: {:.2f}; difference: {:.2f}".format(omega_iter, y_centr, diff))

			# Adding parameters to array
			omega_ycentr_diff_arr.append((omega_pxs, y_centr, diff))

# Variable arrays
omega_data = [point[0] for point in omega_ycentr_diff_arr]
ycentr_data = [point[1] for point in omega_ycentr_diff_arr]
diff_data = [point[2] for point in omega_ycentr_diff_arr]

# Saving data
np.savez('../Data/OYD-R/data_oyd_rolling_90.npz', *[omega_data, ycentr_data, diff_data])