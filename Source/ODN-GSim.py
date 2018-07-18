""" Simulate the difference of the model and centroid points for the global shutter camera, depending on meteor velocity
	and background noise value.
"""

# Python 2/3 compatibility
from __future__ import print_function, division, absolute_import

import numpy as np
import SimulationTools as st
import Parameters as par

# Parameters that are used only for this simulation
rolling_shutter = False
show_plots = True

# Number of iterations for each angular velocity value - 
# used to have a better representatin of the actual difference value
n_iter = 10

# Final array with all 4 noise values
noise_arr = []

# Go through all noise and velocity values
for noise in par.noise_scale_arr:

	noise_diff_arr_iter = []

	for omega_iter in par.omega_odn_arr:
		
		# Average of averages array
		diff_arr = []

		# Get average model - centroid point difference for each meteor
		for i in range(n_iter):

			# Get model and centroid coordinates
			time_coordinates, centroid_coordinates, model_coordinates = st.pointsCentroidAndModel(rolling_shutter, par.t_meteor, par.phi, \
            	omega_iter, par.img_x, par.img_y, par.scale, par.fps, par.sigma_x, par.sigma_y, noise, par.offset, par.fit_param, par.show_plots)

			# Compute difference
			diff = st.centroidAverageDifference(centroid_coordinates, model_coordinates)
			
			# Check difference and other parameters
			print("\t{} omega: {:.2f}; noise: {}; average difference: {:.4f};".format(i, omega_iter, noise, diff))

			diff_arr.append(diff)

		# Calculate overall average difference for all meteors
		avg_diff = np.average(diff_arr)
		
		# Check average of averages array and other parameters
		print("omega: {:.2f}; noise: {}; average of average differences: {:.4f};".format(omega_iter, noise, avg_diff))

		# Append the final, average value to the array
		noise_diff_arr_iter.append(avg_diff)

	# Append all lists to one noise array
	noise_arr.append(noise_diff_arr_iter)


# Set noise array names
noise0_arr = noise_arr[0]
noise1_arr = noise_arr[1]
noise2_arr = noise_arr[2]
noise3_arr = noise_arr[3]

# Save data from the simylation as a file
np.savez('../Data/ODN/data_odn_global.npz', *[par.omega_odn_arr, noise0_arr, noise1_arr, noise2_arr, noise3_arr])