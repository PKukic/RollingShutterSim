""" Simulate the difference of the model and centroid points for the rolling shutter camera, depending on meteor velocity
	and background noise value.
"""

# Python 2/3 compatibility
from __future__ import print_function, division, absolute_import
import numpy as np
import SimulationTools as st
import Parameters as par

# Parameters that are used only for this simulation
rolling_shutter = True
show_plots = False

# Number of iterations for each angular velocity value - 
# used to have a better representatin of the actual difference value
n_iter = 10

# Worst case angle
phi = 0

# Final array
noise_diff_arr = []

noise = 20

for omega_iter in par.omega_odn_arr:
	
	# Average of averages array
	diff_arr = []

	#t_meteor = 0.6

	# Get average model - centroid point difference for each meteor
	for i in range(n_iter):

		# Get model and centroid coordinates
		time_coordinates, centroid_coordinates, model_coordinates = st.pointsCentroidAndModel(rolling_shutter, par.t_meteor, phi, \
        	omega_iter, par.img_x, par.img_y, par.scale, par.fps, par.sigma_x, par.sigma_y, noise, par.offset, par.fit_param, show_plots)

		if (time_coordinates, centroid_coordinates, model_coordinates) != (-1, -1, -1): 

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
	noise_diff_arr.append(avg_diff)


# Save data from the simulation as a file
np.savez('../Data/ODN/data_worst_case.npz', *[par.omega_odn_arr, noise_diff_arr])