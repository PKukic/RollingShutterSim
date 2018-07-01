''' Simulate the difference of the model and corrected centroid points with respect to meteor
	angle and angular velocity, for each noise scale level (simulation_type = 'temporal').
'''

# Python 2/3 compatibility
from __future__ import print_function, division, absolute_import
import SimulationTools as st 
import Parameters as par
import numpy as np 

# Used for testing
show_plots = False

# Initial meteor parameters
time_mark = 'beginning'

# Image center (required for executing the drawPoints function)
x_center = par.img_x/2
y_center = par.img_y/2


phi = par.phi
noise_arr = []

for noise in par.noise_scale_arr:

	noise_diff_arr_iter = []

	for omega in par.omega_odn_arr:

		omega_pxs = omega * par.scale

		# Check the meteor's initial parameters
		print('Meteor velocity: {:.2f}'.format(omega_pxs))
		print('Meteor angle: {}'.format(phi))

		# Get duration of meteor (the meteor is crossing the entire image)
		print('Getting time from angle...')
		t_meteor = st.timeFromAngle	(phi, omega, par.img_x, par.img_y, par.scale, par.fps)

		# Find the deceleration parameters
		dec_arr = [1, st.getparam(1, omega, 0.9*omega, t_meteor)]
		print('Deceleration parameters: ', dec_arr)

		# Get time and centroid coordinates from rolling shutter
		print('Simulating rolling shutter meteor...')
		show_plots = False
		rolling_shutter = True
		time_rolling_coordinates, centroid_rolling_coordinates, model_rolling_coordinates = st.pointsCentroidAndModel(rolling_shutter, t_meteor, phi, \
			omega, par.img_x, par.img_y, par.scale, par.fps, par.sigma_x, par.sigma_y, noise, par.offset, dec_arr, show_plots)

		# Check if meteor is outside of the image
		if (time_rolling_coordinates, centroid_rolling_coordinates, model_rolling_coordinates) != (-1, -1, -1):

			# Correct rolling shutter time coordinates
			print('Correcting time coordinates...')
			time_coordinates_corr = st.timeCorrection(centroid_rolling_coordinates, par.img_y, par.fps, t_meteor, time_mark)

			# Calculate the model points' coordinates
			print('Getting model coordinates...')
			
			model_rolling_coordinates = []

			for i in range(len(time_rolling_coordinates)):

				x_model, y_model = st.drawPoints(time_coordinates_corr[i], x_center, y_center, par.scale, phi, omega, dec_arr, t_meteor)
				model_rolling_coordinates.append((x_model, y_model))


			# Calculate the average difference between the centroid rolling shutter and model meteor points
			print('Calculating average difference between model and centroid rolling shutter points...')
			diff_avg = st.centroidAverageDifference(centroid_rolling_coordinates, model_rolling_coordinates)

			print('Average difference between centroid global and centroid rolling shutter points: {:.2f} [px]'.format(diff_avg))
			
			# print('Difference between centroid global and centroid rolling shutter points for each frame: ')
			# for i in range(len(centroid_rolling_coordinates)):
				# print(st.centroidDifference(centroid_rolling_coordinates[i], centroid_global_coordinates[i]))

			noise_diff_arr_iter.append(diff_avg)

		else:
			print("Meteor is outside of the image!")


	noise_arr.append(noise_diff_arr_iter)


# Set noise array names
noise0_arr = noise_arr[0]
noise1_arr = noise_arr[1]
noise2_arr = noise_arr[2]
noise3_arr = noise_arr[3]

# Save data from the simulation as a file
np.savez('../Data/ODN/data_odn_temp_rolling.npz', *[par.omega_odn_arr, noise0_arr, noise1_arr, noise2_arr, noise3_arr])