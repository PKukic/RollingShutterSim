""" Simulate the difference between the model and centroid meteor points depending on
	meteor velocity and background noise value, both for global and rolling shutter
"""

# Python 2/3 compatibility
from __future__ import print_function, division, absolute_import
import numpy as np
import SimulationTools as st
import Parameters as par

# Parameters used only for this simulation
show_plots = False
rolling_shutter = False

# Number of iterations 
n_iter = 10

# Arrays that contain model-centroid point differences for all velocity and noise values
all_noise_global_arr = []
all_noise_rolling_arr = []

# Go through all noise scale values and meteor velocities
for noise in par.noise_scale_arr:

	noise_rolling_arr_iter = []
	noise_global_arr_iter = []

	for omega in par.omega_odn_arr:

		diff_global_arr = []
		diff_rolling_arr = []

		# Get time that it takes for the meteor to cross the entire image
		t_meteor = st.timeFromAngle(par.phi, omega, par.img_x, par.img_y, par.scale, par.fps)
		print("Omega: {:.2f} t_meteor: {:.2f}".format(omega, t_meteor))

		# Calculate model-centroid point difference for n_iter meteors
		for i in range(n_iter):

			### Calculate difference for global shutter ###
			rolling_shutter = False
			print("{}, rolling: ".format(i), rolling_shutter)

			# Model and centroid coordinates for global shutter
			time_global_coord, centroid_global_coord, model_global_coord = st.pointsCentroidAndModel(rolling_shutter, t_meteor, par.phi, \
            	omega, par.img_x, par.img_y, par.scale, par.fps, par.sigma_x, par.sigma_y, noise, par.offset, show_plots)

			### Calculate difference for rolling shutter ###
			rolling_shutter = True
			print("{}, rolling: ".format(i), rolling_shutter)

			# Rolling shutter model and centroid coordinates
			time_rolling_coord, centroid_rolling_coord, model_rolling_coord = st.pointsCentroidAndModel(rolling_shutter, t_meteor, par.phi, \
            	omega, par.img_x, par.img_y, par.scale, par.fps, par.sigma_x, par.sigma_y, noise, par.offset, show_plots)

			# Check if the meteor is outside of the image
			if (time_rolling_coord, centroid_rolling_coord, model_rolling_coord) != (-1, -1, -1):

				# Apply the correction factor for the rolling shutter centroid coordinates
				centroid_rolling_coord = st.coordinateCorrection(t_meteor, centroid_rolling_coord, par.img_y, par.fps)

				# Calculate average difference of the global shutter model and centroid point coordinates
				diff_global = st.centroidAverageDifference(centroid_global_coord, model_global_coord)

				# Average difference of the rolling shutter model and centroid point coordinates
				diff_rolling = st.centroidAverageDifference(centroid_rolling_coord, model_rolling_coord)

				print("{} Noise: {:.2f} Omega: {:.2f} Diff-R: {:.2f} Diff-G: {:.2f}".format(i, noise, omega, diff_rolling, diff_global))

				# Append the average differences to their arrays
				diff_global_arr.append(diff_global)
				diff_rolling_arr.append(diff_rolling)

		# Calculate overall average differences
		diff_global_avg = np.average(diff_rolling_arr)
		diff_rolling_avg = np.average(diff_global_arr)

		print("Noise: {:.2f} Omega: {:.2f} Diff-G avg: {:.2f} Diff-R avg: {:.2f}".format(noise, omega, diff_global_avg, diff_rolling_avg))

		# Append overall averages to an array
		noise_global_arr_iter.append(diff_global_avg)
		noise_rolling_arr_iter.append(diff_rolling_avg)

	# Append all arrays to final arrays for plotting
	all_noise_global_arr.append(noise_global_arr_iter)
	all_noise_rolling_arr.append(noise_rolling_arr_iter)



# Set array names
noise0_global = all_noise_global_arr[0]
noise1_global = all_noise_global_arr[1]
noise2_global = all_noise_global_arr[2]
noise3_global = all_noise_global_arr[3]

noise0_rolling = all_noise_rolling_arr[0]
noise1_rolling = all_noise_rolling_arr[1]
noise2_rolling = all_noise_rolling_arr[2]
noise3_rolling = all_noise_rolling_arr[3]

# Save the data obtained by the simulation as a file
np.savez('../Data/data_odn_global_rolling.npz', *[par.omega_odn_arr, noise0_global, noise1_global, noise2_global, \
	noise3_global, noise0_rolling, noise1_rolling, noise2_rolling, noise3_rolling])