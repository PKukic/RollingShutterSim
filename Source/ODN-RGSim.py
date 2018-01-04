""" Simulation of a meteor captured by a rolling shutter camera.
"""

# Python 2/3 compatibility
from __future__ import print_function, division, absolute_import
import numpy as np

import SimulationTools as st
import Parameters as par


# Customized parameters
show_plots = False
rolling_shutter = False

### Difference as a function of angular velocity and noise scale ###

n_iter = 10

all_noise_global_arr = []
all_noise_rolling_arr = []

for noise in par.noise_scale_arr:

	noise_rolling_arr = []
	noise_global_arr = []

	for omega_iter in par.omega_odn_arr:

		diff_global_arr = []
		diff_rolling_arr = []

		for i in range(n_iter):

			t_meteor = st.timeFromAngle(par.phi, omega_iter, par.img_x, par.img_y, par.scale, par.fps)

			print("{} t_meteor: {:.2f}".format(i, t_meteor))
			rolling_shutter = not(rolling_shutter)

			# Model and centroid coordinates for global shutter
			centroid_global_coord, model_global_coord = st.pointsCentroidAndModel(rolling_shutter, t_meteor, par.phi, \
            	omega_iter, par.img_x, par.img_y, par.scale, par.fps, par.sigma_x, par.sigma_y, noise, par.offset, show_plots)

			rolling_shutter = not(rolling_shutter)

			# Rolling shutter model and centroid coordinates
			centroid_rolling_coord, model_rolling_coord = st.pointsCentroidAndModel(rolling_shutter, t_meteor, par.phi, \
            	omega_iter, par.img_x, par.img_y, par.scale, par.fps, par.sigma_x, par.sigma_y, noise, par.offset, show_plots)

			if centroid_rolling_coord != -1 and model_rolling_coord != -1:

				# Apply the correction factor for the rolling shutter centroid coordinates
				centroid_rolling_coord = st.coordinateCorrection(t_meteor, centroid_rolling_coord, par.img_y, par.fps)

				# Calculate average difference of the global shutter coordinates
				diff_global = st.centroidAverageDifference(centroid_global_coord, centroid_rolling_coord)

				# Average difference of the rolling shutter coordinates
				diff_rolling = st.centroidAverageDifference(centroid_rolling_coord, model_rolling_coord)

				print("{} Noise scale: {:.2f} Omega: {:.2f} Diff-R: {:.2f} Diff-G: {:.2f}".format(i, noise, omega_iter, diff_rolling, diff_global))

				# Append the average differences to their lists
				diff_global_arr.append(diff_global)
				diff_rolling_arr.append(diff_rolling)

		# Calculate average of lists
		diff_global_avg = np.average(diff_rolling_arr)
		diff_rolling_avg = np.average(diff_global_arr)

		# Append average to list that contains differences of all angular velocities
		noise_global_arr.append(diff_global_avg)
		noise_rolling_arr.append(diff_rolling_avg)

	# Append arrays to final arrays for plotting
	all_noise_global_arr.append(noise_global_arr)
	all_noise_rolling_arr.append(noise_rolling_arr)



# Naming arrays for plotting
noise0_global = all_noise_global_arr[0]
noise1_global = all_noise_global_arr[1]
noise2_global = all_noise_global_arr[2]
noise3_global = all_noise_global_arr[3]

noise0_rolling = all_noise_rolling_arr[0]
noise1_rolling = all_noise_rolling_arr[1]
noise2_rolling = all_noise_rolling_arr[2]
noise3_rolling = all_noise_rolling_arr[3]

# Saving data file
np.savez('../Data/data_odn_global_rolling.npz', *[par.omega_odn_arr, noise0_global, noise1_global, noise2_global, \
	noise3_global, noise0_rolling, noise1_rolling, noise2_rolling, noise3_rolling])