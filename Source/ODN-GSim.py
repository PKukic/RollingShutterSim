""" Simulation of a meteor captured by a rolling shutter camera.
"""

# Python 2/3 compatibility
from __future__ import print_function, division, absolute_import
import numpy as np

import SimulationTools as st
import Parameters as par

# Customised parameters
rolling_shutter = False
show_plots = True

### Difference as a function of angular velocity and noise scale ###

# Number of iterations
n_iter = 10

# Final array with all 4 noise values
noise_arr = []

for noise in par.noise_scale_arr:

	noise_diff_arr = []

	for omega_iter in par.omega_odn_arr:
		
		# Average of averages array
		diff_avg_arr = []

		for i in range(n_iter):

			# Get model and centroid coordinates
			centroid_coordinates, model_coordinates = st.pointsCentroidAndModel(rolling_shutter, par.t_meteor, par.phi, \
            	omega_iter, par.img_x, par.img_y, par.scale, par.fps, par.sigma_x, par.sigma_y, noise, par.offset, par.show_plots)

			# Compute difference
			diff = st.centroidAverageDifference(centroid_coordinates, model_coordinates)
			
			# Check difference and other parameters
			print("\t{} omega: {:.2f}; noise: {}; average difference: {:.4f};".format(i, omega_iter, noise, diff))

			diff_avg_arr.append(diff)


		# Compute overall average of the averages array
		avg = np.average(diff_avg_arr)
		
		# Check average of averages array and other parameters
		print("omega: {:.2f}; noise: {}; average of average differences: {:.4f};".format(omega_iter, noise, avg))

		noise_diff_arr.append(avg)

	# Append all values to the final noise array
	noise_arr.append(noise_diff_arr)


# Naming noise arrays
noise0_arr = noise_arr[0]
noise1_arr = noise_arr[1]
noise2_arr = noise_arr[2]
noise3_arr = noise_arr[3]

# Saving file
np.savez('../Data/data_noise_diff_global.npz', *[par.omega_odn_arr, noise0_arr, noise1_arr, noise2_arr, noise3_arr])