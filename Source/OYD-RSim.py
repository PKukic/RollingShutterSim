""" Simulate the meteor-centroid point difference [diff] depending on the meteor velocity [omega]
	and the Y coordinate of the centroid [ycentr].
"""


# Python 2/3 compatibility
from __future__ import print_function, division, absolute_import
import numpy as np
import SimulationTools as st
import Parameters as par

# Specific parameters used only for this simulation
rolling_shutter = True
show_plots = False

# Meteor angle
phi = 45

# Final data array
omega_ycentr_diff_arr = []

# Go through all velocites
for omega_iter in par.omega_oyd_arr:

	# Get meteor duration (meteor is crossing the entire image)
	t_meteor = st.timeFromAngle(par.phi, omega_iter, par.img_x, par.img_y, par.scale, par.fps)

	print("Meteor duration: {:.2f}".format(t_meteor))

	# Get centroid and model coordinates
	time_coordinates, centroid_coordinates, model_coordinates = st.pointsCentroidAndModel(rolling_shutter, t_meteor, phi, \
        omega_iter, par.img_x, par.img_y, par.scale, par.fps, par.sigma_x, par.sigma_y, par.noise_scale, par.offset, par.fit_param, show_plots)

	# Check if the meteor is outside of the image
	if (time_coordinates, centroid_coordinates, model_coordinates) != (-1, -1, -1):

		# Size of frame array
		frame_num_range = len(centroid_coordinates)
		print("Number of frames: {}".format(frame_num_range))

		# Go through each frame in the meteor
		for frame_num in range(frame_num_range):

			# Calculate model-centroid point difference
			diff = st.centroidDifference(centroid_coordinates[frame_num], model_coordinates[frame_num])

			# Get centroid Y coordinate
			y_centr = centroid_coordinates[frame_num][1]

			# Get meteor velocity in [px/s]
			omega_pxs = omega_iter * par.scale

			# Check all the parameters
			print("Ang. velocity: {:.2f}; Y centroid coordinate: {:.2f}; difference: {:.2f}".format(omega_iter, y_centr, diff))

			# Add all the parameters to the final data array
			omega_ycentr_diff_arr.append((omega_pxs, y_centr, diff))

#  Set data array names
omega_data = [point[0] for point in omega_ycentr_diff_arr]
ycentr_data = [point[1] for point in omega_ycentr_diff_arr]
diff_data = [point[2] for point in omega_ycentr_diff_arr]

# Save the data as a file
np.savez('../Data/OYD-R/data_oyd_rolling_{}.npz'.format(phi), *[omega_data, ycentr_data, diff_data])